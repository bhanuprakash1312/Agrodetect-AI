import type React from "react"
import { Microscope, Leaf, Brain, Zap, CheckCircle2 } from "lucide-react"
import { Progress } from "@/components/ui/progress"

interface LoadingSpinnerProps {
  currentStep: number
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ currentStep }) => {
  const analysisSteps = [
    {
      icon: Microscope,
      text: "Initializing AI models",
      description: "Loading neural networks and preprocessing algorithms",
      shortDescription: "Loading AI models",
      duration: 800,
    },
    {
      icon: Leaf,
      text: "Analyzing image quality",
      description: "Checking resolution, lighting, and focus parameters",
      shortDescription: "Checking image quality",
      duration: 1000,
    },
    {
      icon: Brain,
      text: "Detecting plant features",
      description: "Identifying leaves, stems, and potential disease markers",
      shortDescription: "Detecting features",
      duration: 1200,
    },
    {
      icon: Zap,
      text: "Pattern recognition",
      description: "Comparing against disease database and symptom patterns",
      shortDescription: "Pattern matching",
      duration: 1000,
    },
    {
      icon: CheckCircle2,
      text: "Generating diagnosis",
      description: "Finalizing results and treatment recommendations",
      shortDescription: "Creating diagnosis",
      duration: 600,
    },
  ]

  const getStepStatus = (stepIndex: number) => {
    if (stepIndex < currentStep) return "completed"
    if (stepIndex === currentStep) return "active"
    return "pending"
  }

  const overallProgress = Math.min((currentStep / (analysisSteps.length - 1)) * 100, 100)

  return (
    <div className="flex flex-col items-center justify-center py-6 sm:py-8">
      {/* Main Spinner */}
      <div className="relative mb-6 sm:mb-8">
        {/* Outer rotating ring */}
        <div className="w-20 h-20 sm:w-24 sm:h-24 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin"></div>

        {/* Middle ring */}
        <div
          className="absolute inset-2 w-12 h-12 sm:w-16 sm:h-16 border-4 border-gray-100 border-b-purple-600 rounded-full animate-spin"
          style={{ animationDirection: "reverse", animationDuration: "2s" }}
        ></div>

        {/* Inner core */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
            {currentStep < analysisSteps.length && (() => {
              const Icon = analysisSteps[currentStep].icon
              return <Icon className="w-4 h-4 text-white" />
            })()}
          </div>
        </div>

        {/* Floating particles */}
        <div className="absolute -top-1 sm:-top-2 -right-1 sm:-right-2 w-2 sm:w-3 h-2 sm:h-3 bg-yellow-400 rounded-full animate-bounce"></div>
        <div
          className="absolute -bottom-1 sm:-bottom-2 -left-1 sm:-left-2 w-1.5 sm:w-2 h-1.5 sm:h-2 bg-green-400 rounded-full animate-bounce"
          style={{ animationDelay: "0.5s" }}
        ></div>
        <div
          className="absolute top-1/2 -left-2 sm:-left-4 w-1.5 sm:w-2 h-1.5 sm:h-2 bg-blue-400 rounded-full animate-bounce"
          style={{ animationDelay: "1s" }}
        ></div>
      </div>

      <div className="text-center max-w-2xl w-full px-4">
        <h3 className="text-xl sm:text-2xl font-bold text-gray-900 mb-2">AI Analysis in Progress</h3>
        <p className="text-gray-600 text-base sm:text-lg mb-6 sm:mb-8 leading-relaxed">
          Our advanced machine learning models are examining your crop image
        </p>

        {/* Overall Progress */}
        <div className="mb-6 sm:mb-8">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-700">Analysis Progress</span>
            <span className="text-sm font-medium text-gray-700">{Math.round(overallProgress)}%</span>
          </div>
          <Progress value={overallProgress} className="h-2 sm:h-3" />
        </div>

        {/* Analysis Steps */}
        <div className="space-y-3 sm:space-y-4">
          {analysisSteps.map((step, index) => {
            const Icon = step.icon
            const status = getStepStatus(index)
            
            return (
              <div
                key={index}
                className={`flex items-start gap-3 sm:gap-4 p-3 sm:p-4 rounded-lg sm:rounded-xl border transition-all duration-500 ${
                  status === "completed"
                    ? "bg-green-50 border-green-200"
                    : status === "active"
                    ? "bg-blue-50 border-blue-200 shadow-md"
                    : "bg-gray-50 border-gray-200"
                }`}
              >
                <div
                  className={`w-10 h-10 sm:w-12 sm:h-12 rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-300 ${
                    status === "completed"
                      ? "bg-green-500 text-white"
                      : status === "active"
                      ? "bg-blue-500 text-white animate-pulse"
                      : "bg-gray-300 text-gray-500"
                  }`}
                >
                  {status === "completed" ? (
                    <CheckCircle2 className="w-5 h-5 sm:w-6 sm:h-6" />
                  ) : (
                    <Icon className="w-5 h-5 sm:w-6 sm:h-6" />
                  )}
                </div>
                
                <div className="flex-1 text-left min-w-0">
                  <h4
                    className={`font-semibold mb-1 text-sm sm:text-base ${
                      status === "completed"
                        ? "text-green-800"
                        : status === "active"
                        ? "text-blue-800"
                        : "text-gray-600"
                    }`}
                  >
                    <span className="hidden sm:inline">{step.text}</span>
                    <span className="sm:hidden">{step.shortDescription}</span>
                  </h4>
                  <p
                    className={`text-xs sm:text-sm leading-relaxed ${
                      status === "completed"
                        ? "text-green-600"
                        : status === "active"
                        ? "text-blue-600"
                        : "text-gray-500"
                    }`}
                  >
                    <span className="hidden sm:inline">{step.description}</span>
                    <span className="sm:hidden">{step.shortDescription}</span>
                  </p>
                </div>

                {/* Status indicator */}
                <div className="flex items-center">
                  {status === "active" && (
                    <div className="flex space-x-1">
                      <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-blue-500 rounded-full animate-bounce"></div>
                      <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                      <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                    </div>
                  )}
                  {status === "completed" && (
                    <CheckCircle2 className="w-4 h-4 sm:w-5 sm:h-5 text-green-500" />
                  )}
                </div>
              </div>
            )
          })}
        </div>

        {/* Mobile-optimized Technical Details */}
        <div className="mt-6 sm:mt-8 bg-gray-50 rounded-lg sm:rounded-xl p-3 sm:p-4 border border-gray-200">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 text-xs sm:text-sm">
            <div className="text-center">
              <div className="font-semibold text-gray-900">Model</div>
              <div className="text-gray-600">CNN v2.1</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-gray-900">Accuracy</div>
              <div className="text-gray-600">99.2%</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-gray-900">Classes</div>
              <div className="text-gray-600">50+ diseases</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-gray-900">Processing</div>
              <div className="text-gray-600">Real-time</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
