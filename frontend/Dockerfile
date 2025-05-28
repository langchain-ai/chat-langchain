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

# Create .env file with the environment variables (for build time)
RUN echo "NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}" > .env && \
    echo "API_BASE_URL=${API_BASE_URL}" >> .env && \
    echo "LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}" >> .env

# Copy the rest of the application code
COPY . .

# Print environment variables for debugging
RUN echo "Build environment variables:" && \
    echo "NEXT_PUBLIC_API_URL: ${NEXT_PUBLIC_API_URL}" && \
    echo "API_BASE_URL: ${API_BASE_URL}" && \
    echo "LANGCHAIN_API_KEY available: $(if [ -n \"${LANGCHAIN_API_KEY}\" ]; then echo 'true'; else echo 'false'; fi)"

# Set environment to production and expose variables at runtime
ENV NODE_ENV=production \
    NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    API_BASE_URL=${API_BASE_URL} \
    LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}

# Create .env.local file which takes precedence in Next.js
RUN echo "NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}" > .env.local && \
    echo "API_BASE_URL=${API_BASE_URL}" >> .env.local && \
    echo "LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}" >> .env.local

# Build the Next.js app
RUN yarn build

# Expose the port used by the app
ENV PORT=8080
EXPOSE 8080

# Create runtime script to inject environment variables at container start
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'echo "Runtime environment variables:"' >> /app/start.sh && \
    echo 'echo "NEXT_PUBLIC_API_URL: $NEXT_PUBLIC_API_URL"' >> /app/start.sh && \
    echo 'echo "API_BASE_URL: $API_BASE_URL"' >> /app/start.sh && \
    echo 'echo "LANGCHAIN_API_KEY available: $(if [ -n \"$LANGCHAIN_API_KEY\" ]; then echo true; else echo false; fi)"' >> /app/start.sh && \
    echo 'yarn start -p 8080' >> /app/start.sh && \
    chmod +x /app/start.sh

# Start the server using our script
CMD ["/app/start.sh"]
