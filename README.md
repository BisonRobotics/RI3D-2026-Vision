# RI3D-2026-Vision

## Build for Raspberry Pi 5
### Vision Service
```bash
docker buildx build \
  --platform linux/arm64 \
  -t akafigmo/ri3d-vision:vision \
  --push \
  ./vision
```

### Vision UI (optional)
```bash
docker buildx build \
  --platform linux/arm64 \
  -t akafigmo/ri3d-vision:vision-ui \
  --push \
  ./vision-ui
```