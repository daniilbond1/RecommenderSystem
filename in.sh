#!/bin/bash

gunicorn --bind=0.0.0.0:8888 app:app
