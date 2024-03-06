# Productionizing

This doc will outline steps which are recommended to bring this LLM application to production. 

Bringing LLM applications to production mainly involve preventing/mitigating abuse and guardrails around prompt injection. We'll go over some of your options for these, and suggestions on how to implement them.

> ### We also have in depth security documentation on the main LangChain documentation site. To visit, click [here](https://js.langchain.com/docs/security).

## Abuse

Abuse is a big concern when deploying LLMs to production, mainly due to the cost associated with running the model (or in our case, calling an LLM API). The first and simplest step is to require accounts with some sort of "identity verification". Commonly this is done with an email verification, or for slightly more security, a phone number verification. Adding an account creation step will defer some abuse, and confirming email/phone numbers will make it even harder for users to abuse the system.

However, accounts are not enough to prevent all abuse. The next step is to add rate limiting to your API. This will prevent users from making too many requests in a short period of time. This can be done in a few ways, but the most common is to use a middleware which checks the number of requests a user has made in the last X minutes, often times linked to their account credentials, or if no account is required, then their IP address. If they've made too many, the system can handle it in a variety of ways, often times putting users into a "timeout" for a period of time, or even banning their account/IP if the abuse is severe.

## Prompt Injection

Prompt injection is when a malicious user injects a prompt which tries to make the LLM answer or act in a way you did not originally intend for it to. This sometimes occurs because users want to use the LLM for other reasons than you intended, without having to pay for it themselves. Or the user might be trying to extract information/data they should not have access to, but the LLM does.

The first step you should always do is scoping any permissions your LLM has (eg, database permissions) as tightly as possible. For example, if your LLM only has to answer questions about a database, it should most likely only have write access. Scoping permissions is a good practice in general, but especially important when deploying LLMs.

For other prompt injection prevention methods, the water is a little more murky. We recommend you research your specific use case, and see what the latest security research says about it as the field is changing at a very rapid pace.