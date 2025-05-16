import React, { useState, useRef } from 'react';
import { HiUpload, HiCheck, HiPlay, HiX, HiUserAdd } from 'react-icons/hi';
import PlayerSelectionFlow from './PlayerSelectionFlow';
import Logo from './Logo';

const VideoUpload = () => {
  const [uploadState, setUploadState] = useState('initial'); // initial, uploading, selecting, processing, complete
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [processingSteps, setProcessingSteps] = useState([
    { name: 'Court Detection', status: 'pending', progress: 0 },
    { name: 'Ball Tracking', status: 'pending', progress: 0 },
    { name: 'Player Identification', status: 'pending', progress: 0 },
    { name: 'Bounce Analysis', status: 'pending', progress: 0 },
    { name: 'Generating Visualizations', status: 'pending', progress: 0 }
  ]);
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      simulateUpload(file);
    }
  };

  const openFileSelector = () => {
    fileInputRef.current.click();
  };

  const simulateUpload = (file) => {
    setUploadState('uploading');
    setUploadProgress(0);
    
    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        const newProgress = prev + 5;
        if (newProgress >= 100) {
          clearInterval(interval);
          setUploadState('selecting');
          return 100;
        }
        return newProgress;
      });
    }, 200);
  };

  const handlePlayerSelectionComplete = (selections) => {
    setUploadState('processing');
    simulateProcessing();
  };

  const simulateProcessing = () => {
    // Reset all steps to pending
    setProcessingSteps(steps => 
      steps.map(step => ({ ...step, status: 'pending', progress: 0 }))
    );
    
    // Simulate the steps of processing with realistic timing
    const stepDurations = [3000, 5000, 4000, 3000, 3000]; // milliseconds per step
    let currentStep = 0;
    
    const processStep = () => {
      if (currentStep >= processingSteps.length) {
        setUploadState('complete');
        return;
      }
      
      // Start current step
      setProcessingSteps(steps => {
        const newSteps = [...steps];
        newSteps[currentStep].status = 'processing';
        return newSteps;
      });
      
      // Simulate progress on current step
      const stepInterval = setInterval(() => {
        setProcessingSteps(steps => {
          const newSteps = [...steps];
          const newProgress = Math.min(newSteps[currentStep].progress + 10, 100);
          newSteps[currentStep].progress = newProgress;
          
          // Also update overall progress
          const totalProgress = newSteps.reduce((acc, step, i) => {
            return acc + (step.progress * (i === currentStep ? 0.8 : 1)) / (newSteps.length * 100);
          }, 0) * 100;
          setProcessingProgress(totalProgress);
          
          if (newProgress >= 100) {
            clearInterval(stepInterval);
            newSteps[currentStep].status = 'complete';
            currentStep++;
            setTimeout(processStep, 500); // Start next step after a pause
          }
          
          return newSteps;
        });
      }, stepDurations[currentStep] / 10);
    };
    
    // Start the first step
    processStep();
  };

  const resetUpload = () => {
    setUploadState('initial');
    setUploadProgress(0);
    setProcessingProgress(0);
    setSelectedFile(null);
    setProcessingSteps(steps => 
      steps.map(step => ({ ...step, status: 'pending', progress: 0 }))
    );
  };

  return (
    <div className="flex-1 h-full bg-gray-900 overflow-y-auto">
      <header className="sticky top-0 z-30 bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white">Upload Video</h1>
        </div>
      </header>
      
      {uploadState === 'initial' && (
        <div className="max-w-2xl mx-auto p-6 mt-12">
          <div className="bg-gray-800 rounded-xl p-8 shadow-lg">
            <div className="text-center">
              <Logo size="2xl" className="mx-auto mb-4" />
              <HiUpload className="mx-auto h-16 w-16 text-blue-500" />
              <h2 className="mt-3 text-2xl font-bold text-white">Upload Tennis Match</h2>
              <p className="mt-2 text-gray-400">Upload your tennis match video for analysis</p>
            </div>
            
            <div className="mt-8">
              <div className="border-2 border-dashed border-gray-600 rounded-lg p-12 text-center">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                  accept="video/mp4,video/mov,video/quicktime"
                  className="hidden"
                />
                
                <p className="text-gray-400 mb-6">Supported formats: MP4, MOV (720p or higher recommended)</p>
                <button
                  onClick={openFileSelector}
                  className="mt-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg shadow-lg transition-colors font-medium flex items-center justify-center mx-auto"
                >
                  <HiUpload className="mr-2" />
                  Select Video File
                </button>
              </div>
              
              <div className="mt-8 bg-gray-700 rounded-lg p-6">
                <h3 className="text-white font-bold mb-3">What happens after upload?</h3>
                <ol className="text-sm text-gray-400 space-y-3 list-decimal pl-5">
                  <li>You'll identify players by selecting their ID in each frame</li>
                  <li>Our AI will analyze the match using computer vision</li>
                  <li>You'll receive detailed analytics on player movements, ball tracking, and match statistics</li>
                  <li>The system will generate visualizations like heatmaps and enhanced replay videos</li>
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {uploadState === 'uploading' && (
        <div className="max-w-2xl mx-auto p-6 mt-12">
          <div className="bg-gray-800 rounded-xl p-8 shadow-lg">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-white">Uploading Video</h2>
              <p className="mt-2 text-gray-400">Please wait while your video uploads...</p>
            </div>
            
            <div className="mt-8">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">
                  {selectedFile?.name || 'tennis_match.mp4'}
                </span>
                <span className="text-blue-400">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-blue-500 h-3 rounded-full transition-all duration-300" 
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              
              <div className="mt-4 text-gray-500 text-sm">
                {uploadProgress < 100 ? (
                  <p>This may take a few minutes depending on your connection speed...</p>
                ) : (
                  <div className="flex items-center text-green-500">
                    <HiCheck className="mr-2" /> Upload complete!
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
      
      {uploadState === 'selecting' && (
        <div className="max-w-6xl mx-auto p-4">
          <PlayerSelectionFlow onComplete={handlePlayerSelectionComplete} />
        </div>
      )}
      
      {uploadState === 'processing' && (
        <div className="max-w-2xl mx-auto p-6 mt-12">
          <div className="bg-gray-800 rounded-xl p-8 shadow-lg">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-white">Processing Video</h2>
              <p className="mt-2 text-gray-400">Our AI is analyzing your tennis match...</p>
            </div>
            
            <div className="mt-8">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Overall Progress</span>
                <span className="text-blue-400">{Math.round(processingProgress)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3 mb-6">
                <div 
                  className="bg-blue-500 h-3 rounded-full transition-all duration-300" 
                  style={{ width: `${processingProgress}%` }}
                ></div>
              </div>
              
              {/* Processing Steps */}
              <div className="space-y-4 mt-8">
                {processingSteps.map((step, index) => (
                  <div key={index} className="bg-gray-700 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center">
                        {step.status === 'complete' ? (
                          <div className="rounded-full bg-green-500 p-1 mr-3">
                            <HiCheck className="h-4 w-4 text-white" />
                          </div>
                        ) : step.status === 'processing' ? (
                          <div className="rounded-full border-2 border-blue-500 border-t-transparent animate-spin h-5 w-5 mr-3"></div>
                        ) : (
                          <div className="rounded-full bg-gray-600 p-1 mr-3">
                            <div className="h-4 w-4"></div>
                          </div>
                        )}
                        <span className={`font-medium ${step.status === 'complete' ? 'text-green-400' : step.status === 'processing' ? 'text-white' : 'text-gray-400'}`}>
                          {step.name}
                        </span>
                      </div>
                      <span className="text-sm text-gray-400">{step.progress}%</span>
                    </div>
                    {(step.status === 'processing' || step.status === 'complete') && (
                      <div className="w-full bg-gray-600 rounded-full h-1.5">
                        <div 
                          className={`h-1.5 rounded-full transition-all duration-300 ${step.status === 'complete' ? 'bg-green-500' : 'bg-blue-500'}`}
                          style={{ width: `${step.progress}%` }}
                        ></div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
      
      {uploadState === 'complete' && (
        <div className="max-w-2xl mx-auto p-6 mt-12">
          <div className="bg-gray-800 rounded-xl p-8 shadow-lg">
            <div className="text-center">
              <div className="rounded-full bg-green-500 p-3 inline-flex mx-auto">
                <HiCheck className="h-10 w-10 text-white" />
              </div>
              <h2 className="mt-4 text-2xl font-bold text-white">Processing Complete!</h2>
              <p className="mt-2 text-gray-400">Your tennis match has been fully analyzed</p>
            </div>
            
            <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div className="bg-gray-700 rounded-lg p-6 flex flex-col items-center text-center">
                <HiPlay className="h-8 w-8 text-blue-400 mb-3" />
                <h3 className="text-white font-bold">View Results</h3>
                <p className="text-sm text-gray-400 mt-2">View the complete analysis of your match</p>
                <button className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white rounded-lg py-2 transition-colors">
                  Go to Dashboard
                </button>
              </div>
              
              <div className="bg-gray-700 rounded-lg p-6 flex flex-col items-center text-center">
                <HiUpload className="h-8 w-8 text-purple-400 mb-3" />
                <h3 className="text-white font-bold">Upload Another</h3>
                <p className="text-sm text-gray-400 mt-2">Process another tennis match video</p>
                <button 
                  className="mt-4 w-full bg-gray-600 hover:bg-gray-500 text-white rounded-lg py-2 transition-colors"
                  onClick={resetUpload}
                >
                  Start New Upload
                </button>
              </div>
            </div>
            
            <div className="mt-8 bg-blue-500/10 rounded-lg p-6 border border-blue-500/20">
              <h3 className="text-blue-400 font-bold mb-3 flex items-center">
                <HiUserAdd className="mr-2" /> Add more detailed player info
              </h3>
              <p className="text-sm text-gray-300">
                Complete your player profiles by adding names, teams, and additional metadata for better organization and analytics.
              </p>
              <button className="mt-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg py-2 px-4 transition-colors text-sm">
                Edit Player Profiles
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoUpload; 