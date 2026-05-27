/// Shared models for the WhatsApp-style conversation system.
/// Used by both teacher and parent sides.

class ConversationModel {
  final int id;
  final int? otherUserId;
  final String otherUserName;
  final String? lastMessage;
  final DateTime? lastMessageAt;
  final int unreadCount;

  const ConversationModel({
    required this.id,
    required this.otherUserId,
    required this.otherUserName,
    required this.lastMessage,
    required this.lastMessageAt,
    required this.unreadCount,
  });

  factory ConversationModel.fromJson(Map<String, dynamic> j) =>
      ConversationModel(
        id:             (j['id'] as num).toInt(),
        otherUserId:    (j['other_user_id'] as num?)?.toInt(),
        otherUserName:  j['other_user_name'] as String? ?? '—',
        lastMessage:    j['last_message'] as String?,
        lastMessageAt:  j['last_message_at'] != null
            ? DateTime.tryParse(j['last_message_at'] as String)
            : null,
        unreadCount:    (j['unread_count'] as num?)?.toInt() ?? 0,
      );
}

class ChatMessageModel {
  final int id;
  final int senderId;
  final String body;
  final DateTime createdAt;
  final DateTime? readAt;

  const ChatMessageModel({
    required this.id,
    required this.senderId,
    required this.body,
    required this.createdAt,
    this.readAt,
  });

  bool isMine(int myUserId) => senderId == myUserId;

  factory ChatMessageModel.fromJson(Map<String, dynamic> j) =>
      ChatMessageModel(
        id:        (j['id'] as num).toInt(),
        senderId:  (j['sender_id'] as num).toInt(),
        body:      j['body'] as String,
        createdAt: DateTime.parse(j['created_at'] as String),
        readAt:    j['read_at'] != null
            ? DateTime.tryParse(j['read_at'] as String)
            : null,
      );
}
