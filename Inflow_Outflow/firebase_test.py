'''This program pushes information to firebase'''
# in terminal: pip3 install firebase-admin
# download firebase key added to git_ignore (I'll show you how to get it- needs to be added to gitignore)
import firebase_admin as fb 
from firebase_admin import credentials, firestore

cred = credentials.Certificate("./moc-lots-firebase-adminsdk-pt2o1-1ca13da1f0.json")
default_app = fb.initialize_app(cred)
db = firestore.client()

doc_ref = db.collection(u'test').document(u'VA')
doc_ref.update({
    u'space_avail': "15"
})

print(" database updated")