export const API_URL = "https://dac-healthprice-api.onrender.com";

export const getToken  = () => sessionStorage.getItem("dac_token");
export const getRole   = () => sessionStorage.getItem("dac_role");
export const getUser   = () => sessionStorage.getItem("dac_user");

export function clearAuth() {
  sessionStorage.removeItem("dac_token");
  sessionStorage.removeItem("dac_role");
  sessionStorage.removeItem("dac_user");
}

export function authFetch(url, options = {}) {
  const token = getToken();
  return fetch(url, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });
}
