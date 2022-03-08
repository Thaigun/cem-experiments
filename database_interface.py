import firebase_admin
from firebase_admin import credentials, db
from dotenv import dotenv_values
import json


class DatabaseInterface:
    def __init__(self, name):
        try:
            self.firebase_app = firebase_admin.get_app(name)
        except ValueError:
            config = dotenv_values('.env')
            options = {
                'databaseURL': config['FIREBASE_DATABASE_URL']
            }
            cred = credentials.Certificate('cert/' + config['FIREBASE_KEY_FILE'])
            self.firebase_app = firebase_admin.initialize_app(cred, options, name)
        self.database = db.reference('/experiments', self.firebase_app)

    
    def save_new_entry(self, data_entry):
        self.database.push(data_entry)

    
    def fetch_all_entries(self):
        pass

    
    def fetch_entry_by_id(self, entry_id):
        pass
