"use client";

import { Fragment } from "react";
import { List, ListItem, Spacer, Text, Button } from "@chakra-ui/react";
import { PlusSquareIcon, DeleteIcon } from "@chakra-ui/icons";

import { ThreadListProps } from "../hooks/useThreadList";
import { useThread } from "../hooks/useThread";

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
        <PlusSquareIcon color={"white"} marginRight={"4px"} />
        <Text color={"white"}>New chat</Text>
      </Button>
      <Text color={"white"} marginTop={"24px"} fontWeight={"semibold"}>
        Previous chats
      </Text>
      <List>
        {props.threads?.map((thread, idx) => (
          <Fragment key={thread.thread_id}>
            <ListItem
              role="group"
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
              position={"relative"}
            >
              <Text color={"white"} style={{ width: "100%" }}>
                {(thread.metadata?.["name"] as string) ?? "New chat"}
              </Text>
              <DeleteIcon
                display={"none"}
                _groupHover={{ display: "block" }}
                position={"absolute"}
                right={"12px"}
                bottom={"18px"}
                cursor={"pointer"}
                color={"grey"}
                height={"14px"}
                width={"14px"}
                onPointerDown={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  if (confirm("Delete chat?")) {
                    props.deleteChat(thread.thread_id);
                  }
                }}
              />
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
