
import firebase from 'firebase/app'
import 'firebase/auth'
import 'firebase/firestore'

import { throttle }  from './utils'

const app = firebase.initializeApp({ 
	apiKey: "AIzaSyCS-UQYuYBhXmRODDFQQOMUTJlOWEjzafg",
	authDomain: "nordicjs-emojiwall.firebaseapp.com",
	databaseURL: "https://nordicjs-emojiwall.firebaseio.com",
	projectId: "nordicjs-emojiwall",
	storageBucket: "",
	messagingSenderId: "495162488153",
	appId: "1:495162488153:web:efd7eae80e67c8cabf4848"
})

export const getAPI = async () => {
	const { user } = await firebase.auth().signInAnonymously()
	const db = firebase.firestore()
	
	return {
		setEmoji: throttle((emoji) => db.collection('emojis').doc(user.uid).set({ emoji }), 500),
		getEmojiStream: (listener) => db.collection('emojis').onSnapshot(listener),
		userId: user.uid
	}
}
