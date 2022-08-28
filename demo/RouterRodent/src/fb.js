import firebase from "firebase/app";
import "firebase/firestore";

// Initialize Firebase
const firebaseConfig = {
    apiKey: "AIzaSyCnxGAIxe0Asv_Ly0Ys4CqhE9vMEWOjQ6I",
    authDomain: "esnet-firebase.firebaseapp.com",
    databaseURL: "https://esnet-firebase.firebaseio.com",
    projectId: "esnet-firebase",
    storageBucket: "esnet-firebase.appspot.com",
    messagingSenderId: "533669957655",
    appId: "1:533669957655:web:32b968eb36e9142d"
};

firebase.initializeApp(firebaseConfig);

// Initialize Cloud Firestore through Firebase
export const db = firebase.firestore();
