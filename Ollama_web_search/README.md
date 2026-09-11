```[bash]
docker run -d \
  --name searxng \
  -p 8080:8080 \
  -v ./searxng-config:/etc/searxng \
  -v ./searxng-data:/var/cache/searxng \
  searxng/searxng
```
