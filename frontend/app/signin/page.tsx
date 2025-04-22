import { auth, signIn, signOut } from "@/auth"
 
export default async function SignIn() {
    const session = await auth();
    console.log(session);
    const user = session?.user
    return user ? (
        <div>
            <h1> Welcome {user.name} </h1>

            <form
                action={async () => {
                "use server"
                await signOut();
                }}
            >
                <button>Sign Out</button>
            </form>
        </div>
    ) : 
    (
      <div>
        <h1>You are not authenticated</h1>
        <form
          action={async () => {
            "use server"
            await signIn("google")
          }}
        >
          <button type="submit">Sign in with Google</button>
        </form>
      </div>
    )
  }