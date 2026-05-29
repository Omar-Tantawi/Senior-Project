import 'package:dio/dio.dart';
import 'package:first_try/features/student/presentation/cubit/chatbot_state.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

const String _chatbotBaseUrl = 'http://localhost:8001';
const String _offCurriculumReply = 'هذا السؤال خارج المنهج الذي لديّ.';

class ChatbotCubit extends Cubit<ChatbotState> {
  final Dio _dio = Dio(BaseOptions(
    baseUrl: _chatbotBaseUrl,
    connectTimeout: const Duration(seconds: 15),
    receiveTimeout: const Duration(minutes: 3),
  ));

  ChatbotCubit() : super(const ChatbotIdle([]));

  Future<void> send(String question, {String? grade, String? subject}) async {
    final userMsg = ChatMessage(text: question, isUser: true);
    final updatedMessages = [...state.messages, userMsg];
    emit(ChatbotSending(updatedMessages));

    try {
      final response = await _dio.post('/chat', data: {
        'question': question,
        if (grade != null && grade.isNotEmpty) 'grade': grade,
        if (subject != null && subject.isNotEmpty) 'subject': subject,
      });

      final answer = response.data['answer'] as String;
      final rawSources = response.data['sources'] as List<dynamic>? ?? [];
      final sources = rawSources
          .map((s) => Map<String, dynamic>.from(s as Map))
          .toList();

      final botMsg = ChatMessage(
        text: answer,
        isUser: false,
        sources: sources,
        isOffCurriculum: answer.trim() == _offCurriculumReply || sources.isEmpty,
      );
      emit(ChatbotIdle([...updatedMessages, botMsg]));
    } on DioException catch (e) {
      final errMsg = e.type == DioExceptionType.connectionTimeout ||
              e.type == DioExceptionType.receiveTimeout
          ? 'الاتصال بالمساعد انتهت مهلته. تأكد من تشغيل خادم الذكاء الاصطناعي.'
          : 'تعذر الاتصال بالمساعد. تأكد من تشغيل خادم الذكاء الاصطناعي على المنفذ 8001.';
      emit(ChatbotError(updatedMessages, errMsg));
    }
  }

  void clearChat() => emit(const ChatbotIdle([]));
}
