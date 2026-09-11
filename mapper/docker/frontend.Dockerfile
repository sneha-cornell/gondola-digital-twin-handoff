FROM node:20-bookworm

WORKDIR /app/frontend

COPY frontend/package.json package.json
RUN npm install

COPY frontend .

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
