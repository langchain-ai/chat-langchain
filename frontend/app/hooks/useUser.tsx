import { useEffect, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { getCookie, setCookie } from "../utils/cookies";

export function useUser() {
  const [userId, setUserId] = useState<string>();

  useEffect(() => {
    if (userId) return;

    const userIdCookie = getCookie("clc_user_id");
    if (userIdCookie) {
      setUserId(userIdCookie);
    } else {
      const newUserId = uuidv4();
      setUserId(newUserId);
      setCookie("clc_user_id", newUserId);
    }
  }, []);

  return {
    userId,
  };
}
