APP_NAME=ceksms-core

docker_build:
	docker build -t ceksms-core .

docker_run:
	docker run -it -p 8000:8000 -e PORT=8000 ceksms-core

heroku_login:
	heroku login

heroku_container_login:
	heroku container:login

heroku_push:
	heroku container:push web --app $(APP_NAME)

heroku_release:
	heroku container:release web --app $(APP_NAME)