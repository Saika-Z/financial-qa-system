# --- Part 1 complie phase ---
FROM node:18-alpine AS build-stage
WORKDIR /app

# 1. copy dependencies, reduce build time
COPY packge*.json ./
RUN npm install

# 2. copy source
COPY . .
RUN npm run build


# --- Part 2 run phase ---
# use lightweight nginx
FROM nginx:stable-alpine AS production-stage


# 1. copy dist from complie phase to nginx
COPY --from=build-stage /app/dist /usr/share/nginx/html

# 2. expose port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]