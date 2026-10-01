#!/bin/bash
pkill -9 -f serve-tp4 2>/dev/null
for h in 192.168.1.2 192.168.1.28 192.168.1.31; do ssh -o BatchMode=yes spark@$h "docker rm -f glm53-tf-r0 glm53-tf-r1 glm53-tf-r2 glm53-tf-r3" >/dev/null 2>&1; done
docker rm -f glm53-tf-r0 glm53-tf-r1 glm53-tf-r2 glm53-tf-r3 >/dev/null 2>&1
cd /home/spark/glm53-tf
setsid nohup env CONFIG=config/prod-tp4.env scripts/serve-tp4.sh start >> /home/spark/logs/glm53-tf/swap-tp4.log 2>&1 < /dev/null &
echo LAUNCHED
