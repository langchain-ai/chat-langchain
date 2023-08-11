import Image from 'next/image'
import axios from 'axios'
import Chat from "../app/components/chat"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24 bg-white">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-light text-sm lg:flex">
        <Chat/>
      </div>
    </main>
  )
}


