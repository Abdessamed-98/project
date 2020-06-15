from flask import Blueprint

blueprint = Blueprint(
    'pravda_bouazza',
    __name__,
    url_prefix='/bouazza',
    template_folder='templates',
    static_folder='static'
)
