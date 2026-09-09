// Offline regression checks against the actual TypeScript components.
// Run with Node; uses the existing React/TypeScript dev dependencies.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const ts = require(root + '/node_modules/typescript');
const react = require(root + '/node_modules/react');
const jsx = require(root + '/node_modules/react/jsx-runtime');
const hooks = {...react, useMemo: fn => fn(), useState: v => [typeof v === 'function' ? v() : v, () => {}]};
const Card = () => null;
const players = [{id: 'p1', name: 'Test Player'}, {id: 'p2', name: 'Empty Player'}];
const fixtures = [
 {id:'r1',player_id:'p1',status:'done',createdAt:'2026-09-01T12:00:00Z',shotCount:10,forehandCount:20,backhandCount:30,serveCount:40,inBoundsBounces:8,outBoundsBounces:2},
 {id:'r2',player_id:'p1',status:'done',createdAt:'2026-09-02T12:00:00Z',shotCount:0,forehandCount:90,inBoundsBounces:2,outBoundsBounces:8},
 {id:'r3',player_id:'p1',status:'processing',createdAt:'2026-09-09T12:00:00Z',inBoundsBounces:999,outBoundsBounces:0},
 {id:'r4',player_id:'other',status:'done',createdAt:'2026-09-08T12:00:00Z',inBoundsBounces:999,outBoundsBounces:0},
];
function evaluate(file, imports) {
 const source=ts.transpileModule(fs.readFileSync(root+'/'+file,'utf8'), {compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX}}).outputText;
 const exports={};
 vm.runInNewContext(source,{exports, require: name => {
  if(name==='react')return hooks;
  if(name==='react/jsx-runtime')return jsx;
  if(name in imports)return imports[name];
  return {__esModule:true,default:()=>null};
 }});
 return exports;
}
const Dashboard=evaluate('app/page.tsx', {
 '@/components/dashboard/PlayerCard':{__esModule:true, default:Card},
 '@/contexts/AuthContext':{useAuth:()=>({user:null})},
 '@/lib/hooks/useApiData':{
  useDashboardSummary:()=>({data:null}),usePlayersData:()=>({data:{players}}),useRecordingsData:()=>({data:{recordings:fixtures}})
 }
}).default;
const cards=[];
function walk(node){if(!node||typeof node!=='object')return;if(Array.isArray(node)){node.forEach(walk);return;}if(node.type===Card)cards.push(node.props.player);walk(node.props?.children);}
walk(Dashboard());
assert.equal(cards.length,2);
assert.deepEqual(Array.from(cards[0].metrics,m=>m.label),['In bounds','Recordings','Last']);
assert.deepEqual(Array.from(cards[0].metrics,m=>m.value),[50,2,'Sep 9']);
assert.deepEqual(Array.from(cards[1].metrics,m=>m.value),[0,0,'None yet']);
assert.equal(cards[0].metrics[0].unitSuffix,'%');
console.log('PASS: mismatched legacy counts, completed-only rollup, player isolation, newest pending date, empty roster metrics.');
const Counter=()=>null;
const PlayerCard=evaluate('components/dashboard/PlayerCard.tsx',{
 '@/lib/utils':{playerPhotoProxyUrl:()=>null},
 '@/components/ui/CountUp':{__esModule:true,default:Counter}
}).default;
for(const player of cards){
 const counterValues=[],strings=[];
 function inspect(node){if(typeof node==='string'){strings.push(node);return;}if(!node||typeof node!=='object')return;if(Array.isArray(node)){node.forEach(inspect);return;}if(node.type===Counter)counterValues.push(node.props.to);inspect(node.props?.children);}
 inspect(PlayerCard({player}));
 assert.equal(counterValues.length,2);assert.ok(counterValues.every(v=>typeof v==='number'));
 assert.ok(strings.includes(player.metrics[2].value));
}
console.log('PASS: date and None yet render as text; only numeric values reach CountUp.');
