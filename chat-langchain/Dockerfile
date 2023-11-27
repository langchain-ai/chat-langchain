FROM node:16-buster
RUN mkdir /app
COPY package.json /app/
WORKDIR /app
COPY . ./


RUN npm install
RUN npm run build
EXPOSE 4000
CMD ["npm", "run","start"]
