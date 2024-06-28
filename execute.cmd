docker build -f Dockerfile -t a2c .
docker run -i -t -v "E:\a2c_test\app":/opt/app -w /opt/app --name a2c_container a2c:latest