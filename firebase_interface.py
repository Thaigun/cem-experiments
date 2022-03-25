import firebase_admin
from firebase_admin import credentials, db
from dotenv import dotenv_values


class DatabaseInterface:
    def __init__(self, name, root_path=None):
        try:
            self.firebase_app = firebase_admin.get_app(name)
        except ValueError:
            self.initialize_firebase_app(name)
        if root_path is None:
            config = dotenv_values('.env')
            root_path = config['DB_ROOT']
        self.database = db.reference(root_path, self.firebase_app)


    def initialize_firebase_app(self, name):
        config = dotenv_values('.env')
        options = {
            'databaseURL': config['FIREBASE_DATABASE_URL']
        }
        cred = credentials.Certificate('cert/' + config['FIREBASE_KEY_FILE'])
        self.firebase_app = firebase_admin.initialize_app(cred, options, name)


    def add_new_entry(self, path, data_entry):
        return self.database.child(path).push(data_entry)
    
    def get_child_ref(self, path):
        return self.database.child(path)
    
    def child_exists(self, path):
        return self.database.child(path).get() is not None

    def save_data_in_path(self, path, data):
        self.database.child(path).set(data)
