var a=10;
let b=20;
const c=30;

b=100;
a=200;
//c=300 -> contant doesn't change
console.log(a+b+c);

//if else
if(a=20){
    console.log("this is if condition");
}else{
    console.log("this is else condition");
}
function fruit(item){
    console.log("this is"+ item);
}
console.log("banana");
console.log("apple");

//loops

for(var a=0; a<10; a++){
    console.log(a);
}