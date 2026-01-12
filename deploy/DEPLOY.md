# Deploy to Raspberry Pi 5
Copy `deploy/compose.yaml` to the Pi 5. The `cd` into the directory that houses the file and pull the images with

```bash
docker compose pull
```

After pulling the image, the you can run it with
```bash
docker compose up
```

## Note
I have not set up the services to run on boot. You must manually ssh into the Pi 5 and run `docker compose up` to start the vision service. 

TODO: start the service on boot using chron or systemd.