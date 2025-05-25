# Use official Node.js runtime as a parent image
FROM node:18

# Set the working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

# Set environment to production
ENV NODE_ENV=production

# Copy the rest of the application code
COPY . .

# Build the Next.js app
RUN yarn build

# Expose the port used by the app
ENV PORT=8080
EXPOSE 8080

# Start the server
CMD ["yarn", "start", "-p", "8080"]
