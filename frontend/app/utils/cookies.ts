import Cookies from "js-cookie";

export const getCookie = (name: string): string | undefined => {
  if (typeof window === "undefined") {
    return undefined;
  }
  return Cookies.get(name);
};

export const setCookie = (
  name: string,
  value: string,
  options?: Cookies.CookieAttributes,
): void => {
  if (typeof window === "undefined") {
    return;
  }
  Cookies.set(name, value, {
    expires: 365, // Default to 1 year expiration
    ...(options || {}),
  });
};
