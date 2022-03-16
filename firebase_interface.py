import firebase_admin
from firebase_admin import credentials, db
from dotenv import dotenv_values


class DatabaseInterface:
    def __init__(self, name):
        try:
            self.firebase_app = firebase_admin.get_app(name)
        except ValueError:
            self.initialize_firebase_app(name)
        config = dotenv_values('.env')
        self.database = db.reference(config['DB_ROOT'], self.firebase_app)


    def initialize_firebase_app(self, name):
        config = dotenv_values('.env')
        options = {
            'databaseURL': config['FIREBASE_DATABASE_URL']
        }
        cred = credentials.Certificate('cert/' + config['FIREBASE_KEY_FILE'])
        self.firebase_app = firebase_admin.initialize_app(cred, options, name)


    def save_new_entry(self, path, data_entry):
        return self.database.child(path).push(data_entry)

    
    def get_child_ref(self, path):
        return self.database.child(path)

    
    def fetch_all(self):
        return self.database.get()