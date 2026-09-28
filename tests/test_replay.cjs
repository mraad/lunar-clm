// Exercise replay state transitions without browser automation or packages.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const template = fs.readFileSync(path.join(__dirname, '../lunar_clm/replay.html'), 'utf8');
const source = template.split('<script>')[1].split('</script>')[0];
const before = {x:500,y:30,vx:0,vy:-1,angle:0,fuel:80,time:0,status:'flying',score:0};
const after = {...before,y:28,time:0.2,status:'landed',score:50};
const decision = {proposed:{turn:0,throttle:0},executed:{turn:0,throttle:0},
  intervened:false,answers:null,prompt:'fixture',latency_ms:1};
const episode = {summary:{...after,target:1,seed:0,interventions:0,decisions:1},
  frames:[{before,after,decision}]};
const record = {pilot:{mode:'baseline'},world:{terrain:[[0,20],[1000,20]],
  pads:[[150,240,30,2],[440,560,20,1],[775,825,40,4]]},episodes:[episode]};
const elements = new Map();
const canvas = new Proxy({}, {get:()=>()=>{}});
function element() { return {value:'0',textContent:'',style:{},children:[],
  append(...items){this.children.push(...items)},replaceChildren(){this.children=[]},
  getContext(){return canvas}}; }
const context = vm.createContext({document:{getElementById(id){
  if(!elements.has(id))elements.set(id,element());return elements.get(id);
},createElement:element},requestAnimationFrame(){}});
vm.runInContext(source.replace('/* RECORDING */null', JSON.stringify(record)),context);
assert.equal(elements.get('status').textContent,'FLYING');
assert.equal(elements.get('scrub').max,1);
elements.get('scrub').value='1';elements.get('scrub').oninput();
assert.equal(elements.get('status').textContent,'LANDED');
assert.equal(elements.get('score').textContent,50);
assert.equal(elements.get('play').textContent,'Play');
elements.get('restart').onclick();
assert.equal(elements.get('status').textContent,'FLYING');
assert.equal(elements.get('play').textContent,'Pause');
context.document.getElementById('speed').value='4';
vm.runInContext('tick(0); tick(100);',context);
assert.equal(elements.get('status').textContent,'LANDED');
assert.equal(elements.get('play').textContent,'Play');
console.log('Replay initial state, scrubbing, restart and timed completion: passed');
