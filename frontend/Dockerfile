# Use official Node.js runtime as a parent image
FROM node:18

# Set the working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

# Set build-time variables for Next.js
ARG NEXT_PUBLIC_API_URL
ARG API_BASE_URL
ARG LANGCHAIN_API_KEY

# Set environment to production and expose variables at runtime
ENV NODE_ENV=production \
    NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    API_BASE_URL=${API_BASE_URL} \
    LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}


# Copy the rest of the application code
COPY . .

# Build the Next.js app
RUN yarn build

# Expose the port used by the app
ENV PORT=8080
EXPOSE 8080

# Start the server
CMD ["yarn", "start", "-p", "8080"]
