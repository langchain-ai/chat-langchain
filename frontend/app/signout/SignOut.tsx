"use server";

import { signOut } from "@/auth"

export default async function SignOut() {
    await signOut();
}