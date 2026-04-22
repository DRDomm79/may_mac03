"""
DAC HealthPrice Platform v2.1 — Improved
Fixes: auth, rate limiting, cross-country region validation, input sanitization,
deterministic encoding, champion/challenger, severity clamping, premium bounds,
batch inserts, request ID correlation, error handling.
"""
import os, re, time, uuid, logging, json, hashlib, secrets
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, List
from collections import defaultdict
import joblib, numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator, model_validator
import asyncpg

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),format="%(asctime)s | %(levelname)-7s | %(message)s",datefmt="%H:%M:%S")
log = logging.getLogger("hp")

def _parse_db_url(url):
    if not url: return {}
    url=url.strip()
    b=re.sub(r'^postgresql(\+\w+)?://','',url)
    b=re.sub(r'^postgres://','',b)
    m=re.match(r'^([^:]+):(.+)@([^:/@]+):(\d+)/(.+)$',b)
    if m: return {"user":m[1],"password":m[2],"host":m[3],"port":int(m[4]),"database":m[5]}
    m=re.match(r'^([^:]+):(.+)@([^:/@]+)/(.+)$',b)
    if m: return {"user":m[1],"password":m[2],"host":m[3],"port":5432,"database":m[4]}
    return {}

_db_p=_parse_db_url(os.getenv("DATABASE_URL",""))
MODEL_DIR=os.getenv("MODEL_DIR","models")
ALLOWED_ORIGINS=os.getenv("ALLOWED_ORIGINS","*").split(",")
ADMIN_KEY=os.getenv("ADMIN_API_KEY","")
CF_SECRET=os.getenv("CF_SECRET_TOKEN","")  # Cloudflare secret header — set in Render dashboard
MAX_BODY=int(os.getenv("MAX_BODY_BYTES","65536"))  # 64KB default
GROQ_API_KEY=os.getenv("GROQ_API_KEY","")  # Legacy — kept for fallback
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY","")  # For AI Lab — from console.anthropic.com
db_pool=None; models={}; model_version="v1.0.0"

# Rate limiter
_buckets=defaultdict(lambda:{"t":30,"last":time.monotonic()})
RL=int(os.getenv("RATE_LIMIT_PER_MIN","30"))
def _rl(ip):
    now=time.monotonic();b=_buckets[ip];b["t"]=min(RL,b["t"]+(now-b["last"])*(RL/60));b["last"]=now
    if b["t"]>=1: b["t"]-=1; return True
    return False

async def verify_admin(x_api_key:str=Header(None)):
    if not ADMIN_KEY: raise HTTPException(503,"Admin not configured — set ADMIN_API_KEY env var")
    if x_api_key!=ADMIN_KEY: raise HTTPException(403,"Invalid API key")

VALID_GENDERS=["Male","Female","Other"]
VALID_SMOKING=["Never","Former","Current"]
VALID_EXERCISE=["Sedentary","Light","Moderate","Active"]
VALID_OCC=["Office/Desk","Retail/Service","Healthcare","Manual Labor","Industrial/High-Risk"]
VALID_PE=frozenset(["None","Hypertension","Diabetes","Heart Disease","Asthma/COPD","Cancer (remission)","Kidney Disease","Liver Disease","Obesity","Mental Health"])
CTY_REG={"cambodia":{"Phnom Penh","Siem Reap","Battambang","Sihanoukville","Kampong Cham","Rural Areas"},"vietnam":{"Ho Chi Minh City","Hanoi","Da Nang","Can Tho","Hai Phong","Rural Areas"}}
CTY_REG_L={"cambodia":["Phnom Penh","Siem Reap","Battambang","Sihanoukville","Kampong Cham","Rural Areas"],"vietnam":["Ho Chi Minh City","Hanoi","Da Nang","Can Tho","Hai Phong","Rural Areas"]}
REG_F={"Phnom Penh":1.20,"Siem Reap":1.05,"Battambang":0.90,"Sihanoukville":1.10,"Kampong Cham":0.85,"Ho Chi Minh City":1.25,"Hanoi":1.20,"Da Nang":1.05,"Can Tho":0.90,"Hai Phong":0.95,"Rural Areas":0.75}
G_ENC={"Male":0,"Female":1,"Other":2}
S_ENC={"Never":0,"Former":1,"Current":2}
E_ENC={"Sedentary":0,"Light":1,"Moderate":2,"Active":3}
O_ENC={"Office/Desk":0,"Retail/Service":1,"Healthcare":2,"Manual Labor":3,"Industrial/High-Risk":4}
R_ENC={r:i for i,r in enumerate(["Phnom Penh","Siem Reap","Battambang","Sihanoukville","Kampong Cham","Rural Areas","Ho Chi Minh City","Hanoi","Da Nang","Can Tho","Hai Phong"])}
COV={"ipd":{"name":"IPD Hospital Reimbursement","core":True,"load":0.30},"opd":{"name":"OPD Rider","core":False,"load":0.25},"dental":{"name":"Dental Rider","core":False,"load":0.20},"maternity":{"name":"Maternity Rider","core":False,"load":0.25}}
TIERS={"Bronze":{"annual_limit":15000,"room":"General Ward","surgery_limit":5000,"icu_days":3,"deductible":500},"Silver":{"annual_limit":40000,"room":"Semi-Private","surgery_limit":15000,"icu_days":7,"deductible":250},"Gold":{"annual_limit":80000,"room":"Private Room","surgery_limit":40000,"icu_days":14,"deductible":100},"Platinum":{"annual_limit":150000,"room":"Private Suite","surgery_limit":80000,"icu_days":30,"deductible":0}}
T_F={"Bronze":0.70,"Silver":1.00,"Gold":1.45,"Platinum":2.10}
P_FLOOR=50; P_CEIL=25000

@asynccontextmanager
async def lifespan(app):
    global db_pool,models,model_version
    for c in ["ipd","opd","dental","maternity"]:
        for t in ["freq","sev"]:
            try: models[f"{c}_{t}"]=joblib.load(os.path.join(MODEL_DIR,f"{c}_{t}.pkl")); log.info(f"Loaded {c}_{t}")
            except Exception as e: log.warning(f"Skip {c}_{t}: {e}")
    try: model_version=joblib.load(os.path.join(MODEL_DIR,"model_meta.pkl")).get("version","v1.0.0")
    except: pass
    log.info(f"Models: {list(models.keys())} ({model_version})")
    if _db_p:
        try: db_pool=await asyncpg.create_pool(host=_db_p["host"],port=_db_p["port"],user=_db_p["user"],password=_db_p["password"],database=_db_p["database"],min_size=1,max_size=5,command_timeout=10,timeout=10,ssl="require"); log.info("DB connected")
        except Exception as e: log.warning(f"DB failed: {e}")
    yield
    if db_pool: await db_pool.close()

app=FastAPI(title="DAC HealthPrice API",version="2.1.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=ALLOWED_ORIGINS,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

@app.middleware("http")
async def mw(request:Request,call_next):
    ip=request.client.host if request.client else "x"
    # Block direct access — only allow requests that came through Cloudflare (skip for AI Lab + health)
    is_ailab = request.url.path.startswith("/api/v2/ailab/")
    is_auth = request.url.path == "/auth/login"
    if CF_SECRET and not is_ailab and not is_auth and request.url.path != "/health" and request.headers.get("X-CF-Secret")!=CF_SECRET:
        return JSONResponse(status_code=403,content={"detail":"Direct API access not permitted. Use the official frontend."})
    # Rate limit
    if not _rl(ip): return JSONResponse(429,{"detail":"Rate limit exceeded"})
    # Body size guard
    cl=request.headers.get("content-length")
    body_limit = 52428800 if is_ailab else MAX_BODY  # 50MB for AI Lab file uploads
    if cl and int(cl)>body_limit: return JSONResponse(413,{"detail":"Payload too large"})
    request.state.rid=str(uuid.uuid4())[:12]
    r=await call_next(request)
    # Security headers
    r.headers["X-Request-ID"]=request.state.rid
    r.headers["X-Content-Type-Options"]="nosniff"
    r.headers["X-Frame-Options"]="DENY"
    r.headers["Strict-Transport-Security"]="max-age=31536000; includeSubDomains"
    r.headers["Referrer-Policy"]="strict-origin-when-cross-origin"
    r.headers["X-Permitted-Cross-Domain-Policies"]="none"
    return r

# ── Staff auth ──────────────────────────────────────────────
STAFF_USERS = {
    "admin":  {"hash": hashlib.sha256("dac2026".encode()).hexdigest(), "role": "admin"},
    "actuary": {"hash": hashlib.sha256("dac2026".encode()).hexdigest(), "role": "actuary"},
    "rotha":  {"hash": hashlib.sha256("rotha123".encode()).hexdigest(), "role": "admin"},
    "poly":   {"hash": hashlib.sha256("poly123".encode()).hexdigest(), "role": "actuary"},
}

@app.post("/auth/login")
async def staff_login(request: Request):
    body = await request.json()
    u = body.get("username", "").strip().lower()
    p = body.get("password", "")
    staff = STAFF_USERS.get(u)
    if not staff or hashlib.sha256(p.encode()).hexdigest() != staff["hash"]:
        raise HTTPException(401, "Invalid username or password")
    token = secrets.token_urlsafe(32)
    return {"access_token": token, "role": staff["role"], "username": u}

class PricingRequest(BaseModel):
    age:int=Field(...,ge=0,le=100); gender:str=Field(...); country:str=Field("cambodia"); region:str=Field(...)
    smoking_status:str=Field("Never"); exercise_frequency:str=Field("Light"); occupation_type:str=Field("Office/Desk")
    preexist_conditions:List[str]=Field(default_factory=lambda:["None"])
    ipd_tier:str=Field("Silver"); family_size:int=Field(1,ge=1,le=10)
    include_opd:bool=False; include_dental:bool=False; include_maternity:bool=False

    @field_validator("gender")
    @classmethod
    def vg(cls,v): v=v.strip().title(); assert v in VALID_GENDERS,f"Must be {VALID_GENDERS}"; return v
    @field_validator("country")
    @classmethod
    def vc(cls,v): v=v.strip().lower(); assert v in CTY_REG,f"Must be {list(CTY_REG)}"; return v
    @field_validator("smoking_status")
    @classmethod
    def vs(cls,v): v=v.strip().title(); assert v in VALID_SMOKING,f"Must be {VALID_SMOKING}"; return v
    @field_validator("exercise_frequency")
    @classmethod
    def ve(cls,v): v=v.strip().title(); assert v in VALID_EXERCISE,f"Must be {VALID_EXERCISE}"; return v
    @field_validator("occupation_type")
    @classmethod
    def vo(cls,v): v=v.strip(); assert v in VALID_OCC,f"Must be {VALID_OCC}"; return v
    @field_validator("ipd_tier")
    @classmethod
    def vt(cls,v): v=v.strip().title(); assert v in TIERS,f"Must be {list(TIERS)}"; return v
    @field_validator("preexist_conditions")
    @classmethod
    def vp(cls,v):
        clean=[c.strip() for c in v if c.strip() in VALID_PE]
        if not clean: clean=["None"]
        if "None" in clean and len(clean)>1: clean=[c for c in clean if c!="None"]
        return clean
    @model_validator(mode="after")
    def vr(self):
        if self.region not in CTY_REG.get(self.country,set()):
            raise ValueError(f"'{self.region}' not valid for {self.country}. Valid: {sorted(CTY_REG[self.country])}")
        return self

def _enc(req):
    return np.array([[req.age,G_ENC.get(req.gender,0),S_ENC.get(req.smoking_status,0),E_ENC.get(req.exercise_frequency,1),O_ENC.get(req.occupation_type,0),R_ENC.get(req.region,0),len([p for p in req.preexist_conditions if p!="None"])]])

def _predict(cov,feat):
    fm,sm=models.get(f"{cov}_freq"),models.get(f"{cov}_sev")
    if fm and sm:
        f=float(np.clip(fm.predict(feat)[0],0.001,20));s=float(np.clip(sm.predict(feat)[0],10,100000));src="ml"
    else:
        f=_fb_f(cov,feat);s=_fb_s(cov,feat);src="fallback"
    return {"frequency":round(f,4),"severity":round(s,2),"expected_annual_cost":round(f*s,2),"source":src}

def _fb_f(c,feat):
    a,g,sm,ex,oc,rg,pe=feat[0];base={"ipd":0.12,"opd":2.5,"dental":0.8,"maternity":0.15}.get(c,0.12)
    return max(0.001,base*(1+max(0,(a-35))*0.008)*[1,1.15,1.40][int(min(sm,2))]*[1.20,1.05,0.90,0.80][int(min(ex,3))]*[0.85,1,1.05,1.15,1.30][int(min(oc,4))]*(1+pe*0.20))

def _fb_s(c,feat):
    a,g,sm,ex,oc,rg,pe=feat[0];base={"ipd":2500,"opd":60,"dental":120,"maternity":3500}.get(c,2500)
    rf=[1.20,1.05,0.90,1.10,0.85,0.75,1.25,1.20,1.05,0.90,0.95];ri=int(min(rg,10))
    return max(10,base*(rf[ri] if ri<len(rf) else 1)*(1+max(0,(a-30))*0.006)*(1+pe*0.15))

def _qid(): return f"HP-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

async def _log_q(qid,inp,res):
    if not db_pool: return
    try: await db_pool.execute("INSERT INTO hp_quote_log(quote_ref,input_json,result_json,model_version)VALUES($1,$2::jsonb,$3::jsonb,$4)",qid,json.dumps(inp,default=str),json.dumps(res,default=str),model_version)
    except Exception as e: log.warning(f"Log fail: {e}")

async def _log_b(qid,req):
    if not db_pool: return
    try: await db_pool.execute("INSERT INTO hp_user_behavior(quote_ref,age,gender,country,region,smoking,exercise,occupation,preexist_count,ipd_tier,include_opd,include_dental,include_maternity,family_size)VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)",qid,req.age,req.gender,req.country,req.region,req.smoking_status,req.exercise_frequency,req.occupation_type,len([p for p in req.preexist_conditions if p!="None"]),req.ipd_tier,req.include_opd,req.include_dental,req.include_maternity,req.family_size)
    except Exception as e: log.warning(f"Beh fail: {e}")

@app.get("/health")
async def health():
    return {"status":"healthy","service":"DAC HealthPrice v2.1","models_loaded":list(models.keys()),"model_version":model_version,"database_connected":db_pool is not None,"countries":list(CTY_REG.keys()),"timestamp":datetime.now(timezone.utc).isoformat()}

@app.post("/api/v2/price")
async def calc(req:PricingRequest,request:Request):
    t0=time.monotonic();feat=_enc(req);qid=_qid();rid=getattr(request.state,"rid","?")
    ipd=_predict("ipd",feat);tier=TIERS[req.ipd_tier];tf=T_F[req.ipd_tier];ld=COV["ipd"]["load"]
    ipd_loaded=round(ipd["expected_annual_cost"]*(1+ld)*tf,2)
    ded_cr=round(tier["deductible"]*0.10,2)
    ipd_prem=round(float(np.clip(ipd_loaded-ded_cr,P_FLOOR,P_CEIL)),2)
    riders={};rtot=0
    for c,inc in [("opd",req.include_opd),("dental",req.include_dental),("maternity",req.include_maternity)]:
        if not inc: continue
        r=_predict(c,feat);rp=round(float(np.clip(r["expected_annual_cost"]*(1+COV[c]["load"]),10,5000)),2)
        riders[c]={"name":COV[c]["name"],"frequency":r["frequency"],"severity":r["severity"],"expected_annual_cost":r["expected_annual_cost"],"loading_pct":COV[c]["load"],"annual_premium":rp,"monthly_premium":round(rp/12,2),"source":r["source"]}
        rtot+=rp
    ff=round(1+(req.family_size-1)*0.65,2);pre_fam=round(ipd_prem+rtot,2)
    total=round(float(np.clip(pre_fam*ff,P_FLOOR,P_CEIL*req.family_size)),2)
    res={"quote_id":qid,"request_id":rid,"country":req.country,"region":req.region,"model_version":model_version,"ipd_tier":req.ipd_tier,"tier_benefits":tier,
        "ipd_core":{"frequency":ipd["frequency"],"severity":ipd["severity"],"expected_annual_cost":ipd["expected_annual_cost"],"loading_pct":ld,"tier_factor":tf,"deductible_credit":ded_cr,"annual_premium":ipd_prem,"monthly_premium":round(ipd_prem/12,2),"source":ipd["source"]},
        "riders":riders,"family_size":req.family_size,"family_factor":ff,"total_before_family":pre_fam,"total_annual_premium":total,"total_monthly_premium":round(total/12,2),
        "risk_profile":{"age":req.age,"gender":req.gender,"smoking":req.smoking_status,"exercise":req.exercise_frequency,"occupation":req.occupation_type,"preexist_conditions":req.preexist_conditions,"preexist_count":len([p for p in req.preexist_conditions if p!="None"])},
        "calculated_at":datetime.now(timezone.utc).isoformat()}
    await _log_q(qid,req.model_dump(),res);await _log_b(qid,req)
    ms=round((time.monotonic()-t0)*1000,1);log.info(f"[{rid}] {qid}|{req.ipd_tier}+{'+'.join(riders)or'none'}|age={req.age}|${total:,.0f}/yr|{ms}ms")
    return res

@app.get("/api/v2/reference")
async def ref():
    return {"countries":{k:{"regions":v} for k,v in CTY_REG_L.items()},"genders":VALID_GENDERS,"smoking":VALID_SMOKING,"exercise":VALID_EXERCISE,"occupations":VALID_OCC,"preexist":sorted(VALID_PE),"tiers":TIERS,"tier_factors":T_F,"coverages":COV,"premium_bounds":{"floor":P_FLOOR,"ceiling":P_CEIL}}

@app.get("/api/v2/countries")
async def ctry(): return {"countries":[{"id":k,"name":k.title(),"regions":v} for k,v in CTY_REG_L.items()]}

@app.get("/api/v2/model-info")
async def mi(): return {"version":model_version,"approach":"Freq-Sev (Poisson+GBR)","models":list(models.keys()),"features":["age","gender","smoking","exercise","occupation","region","preexist"],"coverages":list(COV.keys())}

UPLOAD_DIR=os.getenv("UPLOAD_DIR","/tmp/hp_uploads")
REQ_COLS={"age","gender","smoking","exercise","occupation","region","preexist_count","claim_count","claim_amount"}

@app.post("/api/v2/admin/upload-dataset",dependencies=[Depends(verify_admin)])
async def upload(file:UploadFile=File(...),coverage_type:str=Form("ipd"),description:str=Form(""),auto_retrain:bool=Form(False)):
    import pandas as pd,io
    if coverage_type not in COV: raise HTTPException(400,f"Invalid coverage_type: {list(COV.keys())}")
    if not file.filename.endswith(".csv"): raise HTTPException(400,"CSV only")
    contents=await file.read()
    if len(contents)/(1024*1024)>50: raise HTTPException(400,"Max 50MB")
    try: df=pd.read_csv(io.BytesIO(contents))
    except Exception as e: raise HTTPException(400,f"Parse error: {e}")
    miss=REQ_COLS-set(df.columns)
    if miss: return {"status":"rejected","missing":sorted(miss),"required":sorted(REQ_COLS)}
    q={"rows":len(df),"claim_rate":round((df["claim_count"]>0).mean(),4)}
    os.makedirs(UPLOAD_DIR,exist_ok=True);bid=f"up_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}";sp=os.path.join(UPLOAD_DIR,f"{bid}_{coverage_type}.csv");df.to_csv(sp,index=False)
    dbi=0
    if db_pool:
        try:
            recs=[(coverage_type,int(r["age"]),str(r.get("gender","")),str(r.get("smoking","")),str(r.get("exercise","")),str(r.get("occupation","")),str(r.get("region","")),int(r.get("preexist_count",0)),int(r["claim_count"]),float(r.get("claim_amount",0)),bid) for _,r in df.dropna(subset=["age","claim_count"]).iterrows()]
            async with db_pool.acquire() as c: await c.executemany("INSERT INTO hp_claims(coverage_type,age,gender,smoking,exercise,occupation,region,preexist_count,claim_count,claim_amount,batch_id)VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",recs);dbi=len(recs)
        except Exception as e: log.error(f"Batch insert: {e}")
    rr=None
    if auto_retrain and len(df)>=500: rr=await _retrain(coverage_type,sp,bid)
    return {"status":"accepted","batch_id":bid,"rows":len(df),"inserted":dbi,"quality":q,"retrain":rr}

async def _retrain(cov,path,bid):
    global models,model_version
    import pandas as pd
    from sklearn.linear_model import PoissonRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    df=pd.read_csv(path)
    enc={"gender":G_ENC,"smoking":S_ENC,"exercise":E_ENC,"occupation":O_ENC,"region":R_ENC}
    for col,m in enc.items():
        if col in df.columns: df[col]=df[col].map(m).fillna(0).astype(int)
    feat=["age","gender","smoking","exercise","occupation","region","preexist_count"]
    X=df[feat].values;yf=df["claim_count"].values
    cf=PoissonRegressor(alpha=0.01,max_iter=500);cf.fit(X,yf)
    mask=df["claim_count"]>0
    if mask.sum()<50: return {"status":"insufficient","claimants":int(mask.sum())}
    Xs=X[mask];ys=(df.loc[mask,"claim_amount"]/df.loc[mask,"claim_count"]).values
    cs=GradientBoostingRegressor(n_estimators=150,max_depth=4,learning_rate=0.07,random_state=42);cs.fit(Xs,ys)
    cr2=round(cs.score(Xs,ys),4)
    # Champion comparison
    champ=models.get(f"{cov}_sev");promote=True;champ_r2=None
    if champ:
        try: champ_r2=round(champ.score(Xs,ys),4);promote=cr2>=champ_r2-0.02
        except: pass
    nv=f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
    if promote:
        import shutil
        for t in["freq","sev"]:
            p=os.path.join(MODEL_DIR,f"{cov}_{t}.pkl")
            if os.path.exists(p): shutil.copy2(p,p.replace(".pkl","_bak.pkl"))
        joblib.dump(cf,os.path.join(MODEL_DIR,f"{cov}_freq.pkl"));joblib.dump(cs,os.path.join(MODEL_DIR,f"{cov}_sev.pkl"))
        models[f"{cov}_freq"]=cf;models[f"{cov}_sev"]=cs;model_version=nv
        log.info(f"Promoted {cov} {nv} R²={cr2}")
    return {"status":"promoted" if promote else "rejected","version":nv,"r2":cr2,"champion_r2":champ_r2,"rows":len(df)}

@app.get("/api/v2/admin/dataset-template",dependencies=[Depends(verify_admin)])
async def tpl():
    return PlainTextResponse("age,gender,smoking,exercise,occupation,region,preexist_count,claim_count,claim_amount\n35,Male,Current,Sedentary,Office/Desk,Phnom Penh,0,1,2500.00\n28,Female,Never,Moderate,Healthcare,Hanoi,1,0,0\n55,Male,Former,Light,Manual Labor,Rural Areas,2,2,7800.00",media_type="text/csv",headers={"Content-Disposition":"attachment; filename=claims_template.csv"})

@app.get("/api/v2/admin/upload-history",dependencies=[Depends(verify_admin)])
async def uh():
    if not db_pool: return {"status":"no_db"}
    try:
        rows=await db_pool.fetch("SELECT batch_id,coverage_type,COUNT(*)as rows,MIN(ingested_at)as uploaded_at,ROUND(AVG(claim_amount)::numeric,2)as avg FROM hp_claims WHERE batch_id IS NOT NULL GROUP BY batch_id,coverage_type ORDER BY MIN(ingested_at)DESC LIMIT 20")
        return {"status":"ok","uploads":[{k:(v.isoformat() if hasattr(v,'isoformat') else v) for k,v in dict(r).items()} for r in rows]}
    except Exception as e: return {"status":"error","detail":str(e)}

@app.get("/api/v2/admin/user-behavior",dependencies=[Depends(verify_admin)])
async def user_behavior(limit:int=50):
    """Fetch recent user behavior data (quote inputs) for admin dashboard."""
    if not db_pool: return {"status":"no_db","records":[]}
    try:
        rows=await db_pool.fetch("""
            SELECT quote_ref, created_at, age, gender, country, region, smoking, exercise,
                   occupation, preexist_count, ipd_tier, include_opd, include_dental,
                   include_maternity, family_size
            FROM hp_user_behavior ORDER BY created_at DESC LIMIT $1
        """, min(limit, 200))
        records = []
        for r in rows:
            d = dict(r)
            for k,v in d.items():
                if hasattr(v, 'isoformat'): d[k] = v.isoformat()
            records.append(d)
        # Summary stats
        total = len(records)
        summary = {}
        if total > 0:
            ages = [r["age"] for r in records if r.get("age")]
            summary = {
                "total_quotes": total,
                "avg_age": round(sum(ages)/len(ages),1) if ages else 0,
                "tier_distribution": {},
                "rider_rates": {"opd":0,"dental":0,"maternity":0},
                "smoking_distribution": {},
            }
            for r in records:
                t = r.get("ipd_tier","Unknown")
                summary["tier_distribution"][t] = summary["tier_distribution"].get(t,0)+1
                s = r.get("smoking","Unknown")
                summary["smoking_distribution"][s] = summary["smoking_distribution"].get(s,0)+1
                if r.get("include_opd"): summary["rider_rates"]["opd"]+=1
                if r.get("include_dental"): summary["rider_rates"]["dental"]+=1
                if r.get("include_maternity"): summary["rider_rates"]["maternity"]+=1
            for k in summary["rider_rates"]:
                summary["rider_rates"][k] = round(summary["rider_rates"][k]/total*100,1)
        return {"status":"ok","records":records,"summary":summary}
    except Exception as e: return {"status":"error","detail":str(e)}

# ── Actuarial AI Lab ─────────────────────────────────────────────────────────
import tempfile, base64, io, traceback, subprocess

AILAB_SYSTEM_PROMPT = """You are DAC Actuarial AI Assistant — a specialized AI for actuarial data science.
You help actuaries with: data cleaning, EDA, frequency-severity modeling, pricing, reserving (BEL, RA, CSM), IFRS 17 valuation, and expense allocation.

When the user asks for analysis, respond with a brief explanation followed by ONE ```python ``` code block that does everything.

CRITICAL CODE RULES — follow these EXACTLY or the code will fail:
1. Always load data with: df = pd.read_csv("/tmp/ailab_data/{filename}")
2. For charts, ALWAYS use UNIQUE filenames: plt.savefig("/tmp/ailab_output/chart_01_description.png", dpi=150, bbox_inches="tight") then plt.close()
3. NEVER use plt.show() — it will crash. ONLY use plt.savefig() then plt.close()
4. For multiple charts, number them: chart_01_distribution.png, chart_02_correlation.png, chart_03_model.png, etc.
5. Print ALL results with clear labels using print()
6. Always wrap the entire code in try/except and print the error
7. Use these colors for consistency: ['#0d2b7a','#f5a623','#3b82f6','#10b981','#8b5cf6','#ef4444']
8. Set figure style at the start: plt.style.use('seaborn-v0_8-whitegrid') and sns.set_palette(['#0d2b7a','#f5a623','#3b82f6','#10b981'])

ACTUARIAL MODELING RULES:
- Frequency models: use sklearn PoissonRegressor(alpha=0.01) or statsmodels GLM with Poisson family
- Severity models: use sklearn GammaRegressor or GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.07)
- Expected Annual Cost = E[Frequency] x E[Severity]
- Always show model coefficients/feature importance and interpret them
- For classification metrics: accuracy, precision, recall, F1, AUC-ROC
- For regression metrics: MAE, RMSE, R-squared
- Always split data 80/20 for train/test
- Add actuarial interpretation after showing results

OUTPUT FORMAT:
- Keep explanations SHORT (2-3 sentences before the code)
- Put ALL logic in ONE code block — do not split into multiple blocks
- Print a clear summary at the end of the code

Available data columns will be provided in the user message.
The code will be executed automatically in a Python environment with pandas, numpy, sklearn, statsmodels, matplotlib, seaborn available."""

AILAB_UPLOAD_DIR = "/tmp/ailab_data"
AILAB_OUTPUT_DIR = "/tmp/ailab_output"
_ailab_files: dict = {}

@app.post("/api/v2/ailab/upload")
async def ailab_upload(file: UploadFile = File(...)):
    import pandas as pd
    os.makedirs(AILAB_UPLOAD_DIR, exist_ok=True)
    os.makedirs(AILAB_OUTPUT_DIR, exist_ok=True)
    fname = file.filename or "data.csv"
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(400, "Max file size: 50MB")
    fpath = os.path.join(AILAB_UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(contents)
    meta = {"filename": fname, "path": fpath, "size_bytes": len(contents)}
    try:
        if fname.endswith((".csv", ".CSV")):
            df = pd.read_csv(io.BytesIO(contents))
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"status": "uploaded", "meta": meta, "preview": None}
        meta["rows"] = len(df)
        meta["columns"] = list(df.columns)
        meta["dtypes"] = {col: str(dt) for col, dt in df.dtypes.items()}
        meta["missing"] = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}
        meta["numeric_summary"] = {}
        for col in df.select_dtypes(include=["number"]).columns:
            meta["numeric_summary"][col] = {
                "min": round(float(df[col].min()), 2) if not df[col].isna().all() else None,
                "max": round(float(df[col].max()), 2) if not df[col].isna().all() else None,
                "mean": round(float(df[col].mean()), 2) if not df[col].isna().all() else None,
                "std": round(float(df[col].std()), 2) if not df[col].isna().all() else None,
            }
        _ailab_files[fname] = meta
        preview = df.head(10).fillna("").to_dict(orient="records")
        return {"status": "uploaded", "meta": meta, "preview": preview}
    except Exception as e:
        return {"status": "uploaded_parse_error", "meta": meta, "error": str(e)}

@app.post("/api/v2/ailab/analyze")
async def ailab_analyze(request: Request):
    ai_key = ANTHROPIC_API_KEY or GROQ_API_KEY
    if not ai_key:
        raise HTTPException(503, "AI not configured — set ANTHROPIC_API_KEY env var on Render")
    import httpx
    body = await request.json()
    user_message = body.get("message", "")
    history = body.get("history", [])
    filename = body.get("filename", "")
    data_context = ""
    if filename and filename in _ailab_files:
        m = _ailab_files[filename]
        data_context = f"\n\nUploaded file: {filename}\nRows: {m.get('rows', '?')}\nColumns: {m.get('columns', [])}\nData types: {m.get('dtypes', {})}\nMissing values: {m.get('missing', {})}\nNumeric summary: {json.dumps(m.get('numeric_summary', {}), indent=2)}"
    system_prompt = AILAB_SYSTEM_PROMPT + data_context

    if ANTHROPIC_API_KEY:
        # ── Claude API ──
        messages = []
        for msg in history[-20:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 4096,
                        "system": system_prompt,
                        "messages": messages,
                    },
                )
            data = r.json()
            if "error" in data:
                raise HTTPException(502, f"AI error: {data['error'].get('message', 'Unknown')}")
            ai_response = data["content"][0]["text"]
            return {"response": ai_response, "has_code": "```python" in ai_response}
        except httpx.TimeoutException:
            raise HTTPException(504, "AI response timed out")
        except HTTPException: raise
        except Exception as e:
            log.error(f"AI Lab Claude error: {e}"); raise HTTPException(500, "AI analysis failed")
    else:
        # ── Groq fallback ──
        openai_msgs = [{"role": "system", "content": system_prompt}]
        for msg in history[-20:]:
            openai_msgs.append({"role": msg["role"], "content": msg["content"]})
        openai_msgs.append({"role": "user", "content": user_message})
        try:
            async with httpx.AsyncClient(timeout=45) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": "llama-3.3-70b-versatile", "messages": openai_msgs, "max_tokens": 2000, "temperature": 0.1},
                )
            data = r.json()
            if "error" in data:
                raise HTTPException(502, f"AI error: {data['error'].get('message', 'Unknown')}")
            ai_response = data["choices"][0]["message"]["content"]
            return {"response": ai_response, "has_code": "```python" in ai_response}
        except httpx.TimeoutException:
            raise HTTPException(504, "AI response timed out")
        except HTTPException: raise
        except Exception as e:
            log.error(f"AI Lab Groq error: {e}"); raise HTTPException(500, "AI analysis failed")

@app.post("/api/v2/ailab/execute")
async def ailab_execute(request: Request):
    body = await request.json()
    code = body.get("code", "")
    if not code: raise HTTPException(400, "No code provided")
    os.makedirs(AILAB_OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(AILAB_OUTPUT_DIR):
        try: os.remove(os.path.join(AILAB_OUTPUT_DIR, f))
        except: pass
    wrapped = f"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs('/tmp/ailab_output', exist_ok=True)

try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    print(f"ERROR: {{e}}")
"""
    script_path = os.path.join(tempfile.gettempdir(), "ailab_script.py")
    with open(script_path, "w") as f:
        f.write(wrapped)
    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "MPLBACKEND": "Agg"}
        )
        stdout = result.stdout[:50000]
        stderr = result.stderr[:10000] if result.returncode != 0 else ""
        charts = []
        if os.path.exists(AILAB_OUTPUT_DIR):
            for fname in sorted(os.listdir(AILAB_OUTPUT_DIR)):
                fpath = os.path.join(AILAB_OUTPUT_DIR, fname)
                if fname.endswith((".png", ".jpg", ".svg")):
                    with open(fpath, "rb") as img:
                        b64 = base64.b64encode(img.read()).decode()
                        charts.append({"filename": fname, "data": f"data:image/png;base64,{b64}"})
        return {"stdout": stdout, "stderr": stderr, "returncode": result.returncode, "charts": charts, "success": result.returncode == 0}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Execution timed out (120s limit)", "returncode": 1, "charts": [], "success": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": 1, "charts": [], "success": False}

@app.get("/api/v2/ailab/files")
async def ailab_list_files():
    return {"files": list(_ailab_files.values())}

@app.exception_handler(Exception)
async def err(request:Request,exc:Exception):
    rid=getattr(request.state,"rid","?") if hasattr(request,"state") else "?"
    log.error(f"[{rid}] {exc}",exc_info=True)
    return JSONResponse(500,{"detail":"Internal server error","request_id":rid})
