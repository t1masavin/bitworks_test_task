import dill
from flask import Flask, request
from flask_restful import Resource, Api
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

app = Flask(__name__)
api = Api(app)

model = dill.load(
    open(r'redparam',"rb"))
sk = dill.load(
    open(r'scalerparam',"rb"))

pred = {}

class Pred(Resource):
    def get(self, _id):
        return {_id: pred[_id]}

    def put(self, _id):
        pred[_id] = np.matrix.round( model.predict( sk.transform( pd.read_csv(request.form['data'], sep = ',', delimiter= ';' ) ) ) )
                        
        return {_id: list(pred[_id])}

api.add_resource(Pred, '/<int:_id>', endpoint='Pred')

if __name__ == '__main__':
    app.run(debug=True)
