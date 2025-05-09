"use client";
import Script from "next/script";
export const Hotjar_GoogleAnalytics_Snippet = () => {
    return (
        <head>
            <Script id="hotjar-snippet">
                {`
                    (function(h,o,t,j,a,r){
                        h.hj=h.hj||function(){(h.hj.q=h.hj.q||[]).push(arguments)};
                        h._hjSettings={hjid:6393685,hjsv:6};
                        a=o.getElementsByTagName('head')[0];
                        r=o.createElement('script');r.async=1;
                        r.src=t+h._hjSettings.hjid+j+h._hjSettings.hjsv;
                        a.appendChild(r);
                    })(window,document,'https://static.hotjar.com/c/hotjar-','.js?sv=');
                `}
            </Script>
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-JN1K8T232N"></script>
            <Script id="googleanalytics-snippet">
            {`
                window.dataLayer = window.dataLayer || [];
                function gtag(){dataLayer.push(arguments);}
                gtag('js', new Date());

                gtag('config', 'G-JN1K8T232N');
            `}
            </Script>
        </head>
    );
};