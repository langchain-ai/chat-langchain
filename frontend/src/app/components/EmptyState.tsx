import { MouseEvent } from "react";
import { Heading, Card, CardHeader, Flex, Spacer } from "@chakra-ui/react";
import { useTranslations, useLocale } from 'next-intl';
import { useRouter, usePathname, Link } from '@/src/i18n/routing';

export function EmptyState(props: { onChoice: (question: string) => any }) {
  const handleClick = (e: MouseEvent) => {
    props.onChoice((e.target as HTMLDivElement).innerText);
  };
  const router = useRouter();
  const pathname = usePathname();
  const locale = useLocale();  // 获取当前语言

  const handleLocaleChange = (newLocale: string) => {
      // router.replace(`/${newLocale}${pathname}`); // 确保切换时 URL 正确
      window.location.href =  `/${newLocale}${pathname}`;
    };
 const t = useTranslations('HomePage'); 
  return (
    <div className="rounded flex flex-col items-center max-w-full"  style={{ maxWidth: "800px",   margin: "0 auto" }}>
      <Flex marginTop={"25px"} grow={1} maxWidth={"800px"} width={"100%"} justifyContent={"center"}>
        <Card
          onMouseUp={handleClick}
          width={"48%"}
          backgroundColor={"transparent"}
          _hover={{ backgroundColor: "rgb(78,78,81)" }}
          cursor={"pointer"}
          justifyContent={"center"}
        >
          <CardHeader justifyContent={"center"}>
            <Heading
              fontSize="lg"
              fontWeight={"medium"}
              mb={1}
              color={"gray.200"}
              textAlign={"center"}
            >
              {t('What are the key factors affecting Tesla TSLA stock performance recently')}
            </Heading>
          </CardHeader>
        </Card>
        <Spacer />
        <Card
          onMouseUp={handleClick}
          width={"48%"}
          backgroundColor={"transparent"}
          _hover={{ backgroundColor: "rgb(78,78,81)" }}
          cursor={"pointer"}
          justifyContent={"center"}
        >
          <CardHeader justifyContent={"center"}>
            <Heading
              fontSize="lg"
              fontWeight={"medium"}
              mb={1}
              color={"gray.200"}
              textAlign={"center"}
            >
              {t('Is the current US stock market experiencing a bull run')}
            </Heading>
          </CardHeader>
        </Card>
      </Flex>
      <Flex marginTop={"25px"} grow={1} maxWidth={"800px"} width={"100%"} justifyContent={"center"}>
        <Card
          onMouseUp={handleClick}
          width={"48%"}
          backgroundColor={"transparent"}
          _hover={{ backgroundColor: "rgb(78,78,81)" }}
          cursor={"pointer"}
          justifyContent={"center"}
        >
          <CardHeader justifyContent={"center"}>
            <Heading
              fontSize="lg"
              fontWeight={"medium"}
              mb={1}
              color={"gray.200"}
              textAlign={"center"}
            >
              {t('Can the FOMO fear of missing out in the Chinese stock market continue')}              
            </Heading>
          </CardHeader>
        </Card>
        <Spacer />
        <Card
          onMouseUp={handleClick}
          width={"48%"}
          backgroundColor={"transparent"}
          _hover={{ backgroundColor: "rgb(78,78,81)" }}
          cursor={"pointer"}
          justifyContent={"center"}
        >
          <CardHeader justifyContent={"center"}>
            <Heading
              fontSize="lg"
              fontWeight={"medium"}
              mb={1}
              color={"gray.200"}
              textAlign={"center"}
            >
              {t('What are the recent trends and developments in the Nasdaq')}  
            </Heading>
          </CardHeader>
        </Card>
      </Flex>
    </div>
  );
}
