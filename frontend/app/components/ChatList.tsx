"use client";

import { Fragment } from "react";
import { List, ListItem, Spacer, Text, Button } from "@chakra-ui/react";
import { PlusSquareIcon } from "@chakra-ui/icons";

import { ThreadListProps } from "../hooks/useThreadList";
import { useThread } from "../hooks/useThread";

// TODO: this is super rudimentary and is meant to just test things working end-to-end
// need to replace this with a proper UI
export function ChatList(props: {
  threads: ThreadListProps["threads"];
  enterChat: (id: string | null) => void;
  deleteChat: (id: string) => void;
}) {
  const { currentThread } = useThread();
  return (
    <>
      <Button
        backgroundColor={"rgb(58, 58, 61)"}
        _hover={{ backgroundColor: "rgb(78,78,81)" }}
        onClick={() => props.enterChat(null)}
      >
        <PlusSquareIcon
          color={"white"}
          marginRight={"4px"}
          onClick={() => props.enterChat(null)}
        />
        <Text color={"white"}>New chat</Text>
      </Button>
      <Text color={"white"} marginTop={"24px"} fontWeight={"semibold"}>
        Previous chats
      </Text>
      <List>
        {props.threads?.map((thread, idx) => (
          <Fragment key={thread.thread_id}>
            <ListItem
              onClick={() => props.enterChat(thread.thread_id)}
              style={{ width: "100%" }}
              backgroundColor={
                currentThread?.thread_id === thread.thread_id
                  ? "rgb(78,78,81)"
                  : "rgb(58, 58, 61)"
              }
              _hover={{ backgroundColor: "rgb(78,78,81)" }}
              borderRadius={"8px"}
              paddingLeft={"8px"}
              paddingRight={"8px"}
            >
              <Text color={"white"} style={{ width: "100%" }}>
                {(thread.metadata?.["name"] as string) ?? "New chat"}
              </Text>
            </ListItem>
            {idx !== (props.threads?.length ?? 0 - 1) && (
              <Spacer height={"4px"} />
            )}
          </Fragment>
        ))}
      </List>
    </>
  );
}
