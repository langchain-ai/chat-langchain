import {
  ChatComposerAttachment,
  ChatComposerAttachments,
} from "@langchain/macaw-components/ChatComposer"
import type { ImageAttachment } from "@/lib/types"

interface FilePreviewGridProps {
  files: ImageAttachment[]
  onRemove: (fileId: string) => void
}

export function FilePreviewGrid({ files, onRemove }: FilePreviewGridProps) {
  if (files.length === 0) return null

  return (
    <ChatComposerAttachments aria-label="Attached files" layout="wrap">
      {files.map((file) => {
        const name = file.name || "Attachment"
        const imageSource =
          file.url ||
          (file.base64
            ? `data:${file.mimeType};base64,${file.base64}`
            : undefined)
        const isImage = file.mimeType.startsWith("image/") && imageSource

        return (
          <ChatComposerAttachment
            key={file.id}
            name={name}
            contentType={file.mimeType}
            variant={isImage ? "thumbnail" : "file"}
            preview={
              isImage ? (
                <img
                  src={imageSource}
                  alt={name}
                  className="h-full w-full object-cover"
                />
              ) : undefined
            }
            metadata={
              file.size
                ? `${Math.max(1, Math.round(file.size / 1024))} KB`
                : undefined
            }
            onRemove={() => onRemove(file.id)}
          />
        )
      })}
    </ChatComposerAttachments>
  )
}
