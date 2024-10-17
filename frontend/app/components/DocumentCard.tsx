import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

export type Document = {
  page_content: string;
  metadata: Record<string, any>;
};

export const DocumentCard = ({ document }: { document: Document }) => {
  const description = document.page_content.slice(0, 250);

  return (
    <Card className="md:w-[200px] sm:w-[200px] w-full h-[110px] bg-inherit border-gray-500 flex flex-col">
      <CardHeader className="flex-shrink-0 px-3 pt-2 pb-0">
        <CardTitle className="text-sm font-light text-gray-300 line-clamp-1 overflow-hidden p-[-24px]">
          {document.metadata.title}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col px-3 flex-grow justify-between">
        <p className="text-xs font-light text-gray-400 line-clamp-4 overflow-hidden">
          {description}
        </p>
      </CardContent>
    </Card>
  );
};
