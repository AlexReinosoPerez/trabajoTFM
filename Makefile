# Ruta del docker-compose
COMPOSE_FILE=containers/docker-compose.yml

#  Build de imágenes
build:
	docker compose -f $(COMPOSE_FILE) build

#  Levantar todos los servicios
up:
	docker compose -f $(COMPOSE_FILE) up -d

#  Parar los servicios
down:
	docker compose -f $(COMPOSE_FILE) down

#  Ver logs de todos los servicios
logs:
	docker compose -f $(COMPOSE_FILE) logs -f

#  Streamlit (logs solo de la interfaz)
logs-streamlit:
	docker compose -f $(COMPOSE_FILE) logs -f streamlit

#  Entrenamiento (logs solo del job de entrenamiento)
logs-train:
	docker compose -f $(COMPOSE_FILE) logs -f trainer

#  Entrenar el modelo desde contenedor (vuelve a lanzar solo trainer)
retrain:
	docker compose -f $(COMPOSE_FILE) up --build trainer

#  Eliminar todos los contenedores/paradas
prune:
	docker system prune -af --volumes
