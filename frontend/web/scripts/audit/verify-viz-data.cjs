// Offline regression checks against the actual TypeScript components.
// Run with Node; uses the existing React/TypeScript dev dependencies.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=require('node:path').resolve(__dirname, '../..'), ts=require(root+'/node_modules/typescript'), jsx=require(root+'/node_modules/react/jsx-runtime');
const source=ts.transpileModule(fs.readFileSync(root+'/components/recordings/VizPanel.tsx','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX}}).outputText;
function check(mode,props={}) {
 const exports={};
 vm.runInNewContext(source,{exports,require:name=>{
  if(name==='react/jsx-runtime')return jsx;
  if(name==='react')return {useMemo:fn=>fn(),useEffect:()=>{},useState:v=>[v==='shotMap'?mode:v,()=>{}]};
  return {__esModule:true,default:name,shotMapCounts:()=>({}),spacingCounts:()=>({}),shotMapUnknownCount:()=>0};
 }});
 const result={callouts:[],courts:[]};
 function walk(node){if(!node||typeof node!=='object')return;if(Array.isArray(node)){node.forEach(walk);return;}if(node.type?.name==='CoachingInsight')result.callouts.push(node);if(['../viz/ShotMap','../viz/Spacing','../viz/Coverage'].includes(node.type))result.courts.push(node);walk(node.props?.children);}
 walk(exports.default(props));return result;
}
for(const mode of ['spacing','coverage']){
 assert.equal(check(mode).callouts.length,0,mode+' must hide commentary without real data');
 assert.equal(check(mode).courts.length,0,mode+' must render its unavailable state without real data');
}
assert.equal(check('shotMap').courts[0].props.dots.length,0,'empty recording must render a court with no dots');
assert.equal(check('coverage',{coverageGrid:[[1,0],[0,3]]}).callouts.length,1);
const shots=[{stroke:'forehand',player:1,player_court_x:10,player_court_y:60,ball_court_x:12.5,ball_court_y:60,court_x:12,court_y:20,in:true}];
assert.equal(check('spacing',{shots}).callouts.length,1);
assert.equal(check('spacing',{shots}).courts[0].props.shots.length,1,'real spacing is retained');
assert.equal(check('shotMap',{shots}).courts[0].props.dots.length,1,'real dots are retained');
console.log('PASS: no sample fixtures; missing spacing/coverage commentary hidden, real-data commentary preserved.');
