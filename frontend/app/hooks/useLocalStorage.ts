import { useState } from "react";

type StoredValue = string | string[] | Record<string, string> | null;

export function useLocalStorage(key: string, initialValue: StoredValue) {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      if (typeof window === "undefined") {
        return;
      }

      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.log(error);
      return initialValue;
    }
  });
  const setValue = (value: StoredValue) => {
    try {
      if (typeof window === "undefined") {
        return;
      }

      setStoredValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.log(error);
    }
  };
  return [storedValue, setValue];
}
