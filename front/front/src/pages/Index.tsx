import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Sparkles, AlertCircle, Lightbulb } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import type { PromptResponse, EvaluateAnswerResponse } from "./types";

const Index = () => {
	const [currentQuestion, setCurrentQuestion] = useState("");
	const [answer, setAnswer] = useState("");
	const [evaluation, setEvaluation] = useState<{
		score: number;
		improvements: string[];
		improvedAnswer: string;
	} | null>(null);
	const [isGenerating, setIsGenerating] = useState(false);
	const { toast } = useToast();

  // generar pregunta
	const generateQuestion = async () => {
		console.log("inicio de generate question")
		try {
		setIsGenerating(true);
		setEvaluation(null);
		setAnswer("");

		const res = await fetch("/api/generate", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
			prompt: "hazme una pregunta de entrevista de trabajo",
			system_prompt: "solo muestra la pregunta",
			temperature: 0.7,
			max_tokens: 1000,
			}),
		});

		if (!res.ok) throw new Error("No se pudo generar la pregunta");
		const data: PromptResponse = await res.json();

		setCurrentQuestion((data.response || "").trim());
		if (!data.response) {
			toast({
			title: "Sin contenido",
			description: "El modelo no devolvió texto",
			variant: "destructive",
			});
		}
		} catch (e: any) {
		toast({
			title: "Error",
			description: e?.message || "No se pudo generar la pregunta",
			variant: "destructive",
		});
		} finally {
		setIsGenerating(false);
		}
	};

	// evaluar pregunta
	const handleEvaluate = async () => {
		console.log("inicio de handle Evaluate")
		if (!answer.trim()) {
		toast({
			title: "Respuesta vacía",
			description: "Escribe tu respuesta antes de evaluar",
			variant: "destructive",
		});
		return;
		}
		if (!currentQuestion) {
		toast({
			title: "No hay pregunta",
			description: "Primero genera una pregunta",
			variant: "destructive",
		});
		return;
		}

		try {
			console.log("1")
		// pregunta + respuesta
		const res = await fetch("/api/evaluateAnswer", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
			prompt: `Pregunta: ${currentQuestion}\nRespuesta: ${answer}`,
			system_prompt:
				"eres un evaluador de respuestas de entrevistas. da un puntanje a la respuesta entre 1 -100.da una sugerencia breve. muestra una respuesta sugerida breve. muestra la respuesta sugerida sin decir que es una respuesta sugerida",
				//"eres un evaluador de respuestas de entrevistas de trabajo. muestra sugerencia. no utilices embellecedores de texto",
			temperature: 0.7,
			max_tokens: 1500,
			}),
		});
		console.log("2")

		if (!res.ok) throw new Error("No se pudo evaluar la respuesta");
		const data: EvaluateAnswerResponse = await res.json();
		const score = Math.max(0, Math.min(100, data.response.rating ?? 0));
		const improvedAnswer = (data.response.answer || "").trim();

		setEvaluation({
			score,
			improvements: [],
			improvedAnswer,
		});

		console.log(improvedAnswer)

		toast({
			title: "Respuesta evaluada",
			description: `Tu puntuación: ${score}%`,
		});
		} catch (e: any) {
		toast({
			title: "Error al evaluar",
			description: e?.message || "Intenta nuevamente",
			variant: "destructive",
		});
		}
		console.log("fin de handle Evaluate")
	};


	return (
		<div className="min-h-screen bg-background">
			{/* Hero Section */}
			<div className="relative overflow-hidden">
				<div className="absolute inset-0 bg-gradient-primary opacity-10" />
				<div className="container mx-auto px-4 py-16 md:py-24 relative">
					<div className="max-w-4xl mx-auto text-center space-y-6">
						<div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent text-accent-foreground text-sm font-medium mb-4">
							<Sparkles className="w-4 h-4" />
							Powered by AI
						</div>

						<h1 className="text-4xl md:text-6xl font-bold tracking-tight">
							Practica entrevistas con{" "}
							<span className="bg-gradient-primary bg-clip-text text-transparent">
								Inteligencia Artificial
							</span>
						</h1>

						<p className="text-xl text-muted-foreground max-w-2xl mx-auto">
							Prepárate para tu próxima entrevista de trabajo. Genera preguntas
							realistas, responde y recibe feedback instantáneo con sugerencias
							de mejora.
						</p>
					</div>
				</div>
			</div>

			{/* Main Content */}
			<div className="container mx-auto px-4 pb-16 md:pb-24">
				<div className="max-w-4xl mx-auto space-y-8">
					{/* Generate Question Button */}
					<div className="flex justify-center">
						<Button
							size="lg"
							onClick={generateQuestion}
							disabled={isGenerating}
							className="text-lg px-8 py-6 h-auto bg-gradient-primary hover:opacity-90 transition-opacity"
						>
							{isGenerating ? (
								<>
									<Sparkles className="w-5 h-5 mr-2 animate-spin" />
									Generando pregunta...
								</>
							) : (
								<>
									<Sparkles className="w-5 h-5 mr-2" />
									Generar Pregunta
								</>
							)}
						</Button>
					</div>

					{/* Question Display */}
					{currentQuestion && (
						<Card className="p-6 md:p-8 border-2 animate-in fade-in slide-in-from-bottom-4 duration-500">
							<div className="space-y-4">
								<div className="flex items-center gap-2 text-primary font-semibold">
									<AlertCircle className="w-5 h-5" />
									Pregunta de entrevista
								</div>
								<p className="text-lg md:text-xl font-medium leading-relaxed">
									{currentQuestion}
								</p>
							</div>
						</Card>
					)}

					{/* Answer Textarea */}
					{currentQuestion && (
						<div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500 delay-150">
							<label className="text-sm font-medium">Tu respuesta</label>
							<Textarea
								placeholder="Escribe tu respuesta aquí..."
								value={answer}
								onChange={(e) => setAnswer(e.target.value)}
								className="min-h-[200px] text-base resize-none"
							/>
							<div className="flex justify-end">
								<Button
									onClick={handleEvaluate}
									disabled={!answer.trim()}
									className="px-6"
								>
									Evaluar respuesta
								</Button>
							</div>
						</div>
					)}

					{/* Evaluation Results */}
					{evaluation && (
						<div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
							{/* Score Card */}
							<Card className="p-6 md:p-8 border-2 border-primary bg-gradient-to-br from-primary/5 to-secondary/5">
								<div className="flex items-center gap-4">
									<div className="flex-shrink-0">
										<div className="w-20 h-20 rounded-full bg-gradient-primary flex items-center justify-center">
											<span className="text-3xl font-bold text-white">
												{evaluation.score}%
											</span>
										</div>
									</div>
									<div>
										<h3 className="text-xl font-semibold mb-1">
											{evaluation.score >= 80
												? "¡Excelente respuesta!"
												: evaluation.score >= 60
													? "Buena respuesta"
													: "Puedes mejorar"}
										</h3>
										<p className="text-muted-foreground">
											Puntuación de tu respuesta
										</p>
									</div>
								</div>
							</Card>

							{/* Improvements
							<Card className="p-6 md:p-8">
								<div className="space-y-4">
									<div className="flex items-center gap-2 text-foreground font-semibold">
										<CheckCircle2 className="w-5 h-5 text-success" />
										Puntos a mejorar
									</div>
									<ul className="space-y-2">
										{evaluation.improvements.map((improvement, index) => (
											<li key={index} className="flex items-start gap-2">
												<span className="text-primary mt-1">•</span>
												<span className="text-muted-foreground">
													{improvement}
												</span>
											</li>
										))}
									</ul>
								</div>
							</Card> */}

							{/* Improved Answer */}
							<Card className="p-6 md:p-8 border-2">
								<div className="space-y-4">
									<div className="flex items-center gap-2 text-foreground font-semibold">
										<Lightbulb className="w-5 h-5 text-secondary" />
										Resutados
									</div>
									<p className="text-muted-foreground leading-relaxed whitespace-pre-line">
										{evaluation.improvedAnswer}
									</p>
								</div>
							</Card>
						</div>
					)}
				</div>
			</div>
		</div>
	);
};

export default Index;
