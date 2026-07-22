#!/usr/bin/env bash
# confirma que los servicios LEVANTADOS responden antes de ejecutar un
# capitulo. solo comprueba lo que esta corriendo: arbitra entre "el codigo
# falla" y "el servicio esta caido", igual que el objetivo _smoke arbitra la
# compilacion en latex. no arranca nada; eso es tarea de docker compose.
set -uo pipefail

cd "$(dirname "$0")"
COMPOSE="docker compose -f docker-compose.yml"

# servicios que estan corriendo ahora mismo
running="$($COMPOSE ps --services --status running 2>/dev/null)"

fallo=0
comprobar() {  # $1 nombre de servicio  $2 orden de sondeo
  local svc="$1" probe="$2"
  if ! grep -qx "$svc" <<<"$running"; then
    printf '  %-12s no levantado (omitido)\n' "$svc"
    return
  fi
  if eval "$probe" >/dev/null 2>&1; then
    printf '  %-12s OK\n' "$svc"
  else
    printf '  %-12s LEVANTADO PERO NO RESPONDE\n' "$svc"
    fallo=1
  fi
}

echo "comprobando servicios activos..."
comprobar postgres    "$COMPOSE exec -T postgres pg_isready -U libro -d libro"
comprobar timescaledb "$COMPOSE exec -T timescaledb pg_isready -U libro"
comprobar redis       "$COMPOSE exec -T redis redis-cli ping"
comprobar mongo       "$COMPOSE exec -T mongo mongosh --quiet --eval 'db.runCommand({ping:1})'"
comprobar qdrant      "curl -sf http://localhost:6333/healthz"
comprobar milvus      "curl -sf http://localhost:9091/healthz"

if [ "$fallo" -ne 0 ]; then
  echo "hay servicios caidos; revisa 'docker compose logs'." >&2
  exit 1
fi
echo "todos los servicios activos responden."
