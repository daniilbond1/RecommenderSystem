from flask import Flask

from src.connector import controller

from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.register_blueprint(blueprint=controller)

def main():
    app.run(port=5004)

app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
    main()
