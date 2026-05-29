import 'package:equatable/equatable.dart';

class ChatMessage extends Equatable {
  final String text;
  final bool isUser;
  final List<Map<String, dynamic>> sources;
  final bool isOffCurriculum;

  const ChatMessage({
    required this.text,
    required this.isUser,
    this.sources = const [],
    this.isOffCurriculum = false,
  });

  @override
  List<Object?> get props => [text, isUser, sources, isOffCurriculum];
}

abstract class ChatbotState extends Equatable {
  final List<ChatMessage> messages;
  const ChatbotState(this.messages);
  @override
  List<Object?> get props => [messages];
}

class ChatbotIdle extends ChatbotState {
  const ChatbotIdle(super.messages);
}

class ChatbotSending extends ChatbotState {
  const ChatbotSending(super.messages);
}

class ChatbotError extends ChatbotState {
  final String error;
  const ChatbotError(super.messages, this.error);
  @override
  List<Object?> get props => [messages, error];
}
