// Offline regression checks against the actual TypeScript components.
// Run with Node; uses the existing React/TypeScript dev dependencies.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=require('node:path').resolve(__dirname, '../..'), ts=require(root+'/node_modules/typescript'), jsx=require(root+'/node_modules/react/jsx-runtime');
const source=ts.transpileModule(fs.readFileSync(root+'/components/recordings/VizPanel.tsx','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX}}).outputText;
function check(mode,demo,props={}) {
 const exports={};
 vm.runInNewContext(source,{exports,require:name=>{
  if(name==='react/jsx-runtime')return jsx;
  if(name==='react')return {useMemo:fn=>fn(),useEffect:()=>{},useState:v=>[v==='shotMap'?mode:v,()=>{}]};
  if(name==='@/lib/demo/demoData')return {isDemoMode:()=>demo};
  return {__esModule:true,default:()=>null,SAMPLE_SHOTS:[],SAMPLE_SPACING:[{q:'ideal'}],shotMapCounts:()=>({}),spacingCounts:()=>({}),shotMapUnknownCount:()=>0};
 }});
 const callouts=[];
 function walk(node){if(!node||typeof node!=='object')return;if(Array.isArray(node)){node.forEach(walk);return;}if(node.type?.name==='CoachingInsight')callouts.push(node);walk(node.props?.children);}
 walk(exports.default({recordingStatus:'done',...props}));return callouts;
}
for(const mode of ['spacing','coverage']){
 assert.equal(check(mode,false).length,0,mode+' must hide commentary without real data');
 assert.equal(check(mode,true).length,1,mode+' demo commentary remains');
}
assert.equal(check('coverage',false,{coverageGrid:[[1,0],[0,3]]}).length,1);
assert.equal(check('spacing',false,{shots:[{stroke:'forehand',player:1,player_court_x:10,player_court_y:60,ball_court_x:12.5,ball_court_y:60,court_x:12,court_y:20,in:true}]}).length,1);
console.log('PASS: missing spacing/coverage commentary hidden for real recordings, present in demo, real-data commentary preserved.');
