# Thin wrappers over the commands you would otherwise have to remember.
# Everything runs in Docker, so a clone plus Docker is enough.
.PHONY: help up cloud down logs ingest reingest test eval eval-compare eval-sweep lint shell health metrics clean

COMPOSE := docker compose
APP_PORT ?= 8080

help:            ## show this list
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t22

up:              ## start everything with the local model (first run pulls models)
	./start.sh

cloud:           ## start with the cloud model (needs OPENAI_API_KEY in the shell)
	./start.sh --cloud

down:            ## stop the stack
	./stop.sh

logs:            ## follow the app logs
	$(COMPOSE) logs -f app

ingest:          ## index new or changed documents
	$(COMPOSE) exec -T app python -m app.ingest.run

reingest:        ## re-embed everything (after changing the chunker or the model)
	$(COMPOSE) exec -T app python -m app.ingest.run --force

test:            ## unit + integration tests, in a throwaway container
	./scripts/test.sh

eval:            ## measure retrieval against the golden questions
	$(COMPOSE) exec -T app python -m app.eval.run

eval-compare:    ## same, one technique at a time, against a dense-only baseline
	$(COMPOSE) exec -T app python -m app.eval.run --compare

eval-sweep:      ## tune RELATIVE_CUTOFF: how few sources before quality drops
	$(COMPOSE) exec -T app python -m app.eval.run --sweep

lint:            ## ruff, in a throwaway container
	docker run --rm -v "$(PWD)":/w -w /w python:3.11-slim \
	  sh -c 'pip install -q ruff && ruff check .'

health:          ## is the stack ready?
	@curl -s http://localhost:$(APP_PORT)/health | python3 -m json.tool

metrics:         ## answer rate, latency, questions nothing could answer
	@curl -s http://localhost:$(APP_PORT)/metrics | python3 -m json.tool

shell:           ## a shell inside the app container
	$(COMPOSE) exec app bash
