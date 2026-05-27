import 'package:first_try/features/auth/current_user.dart';
import 'package:first_try/features/parent/presentation/cubit/parent_chat_cubit.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:intl/intl.dart';

class ParentChatScreen extends StatefulWidget {
  final int    conversationId;
  final String otherUserName;

  const ParentChatScreen({
    super.key,
    required this.conversationId,
    required this.otherUserName,
  });

  @override
  State<ParentChatScreen> createState() => _ParentChatScreenState();
}

class _ParentChatScreenState extends State<ParentChatScreen> {
  final _ctrl       = TextEditingController();
  final _scrollCtrl = ScrollController();

  @override
  void dispose() {
    _ctrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollCtrl.hasClients) return;
      _scrollCtrl.animateTo(
        _scrollCtrl.position.maxScrollExtent,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final myUserId = context.currentUserId;
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        titleSpacing: 0,
        title: Row(
          children: [
            CircleAvatar(
              radius: 18,
              backgroundColor: cs.primaryContainer,
              child: Text(
                widget.otherUserName.isNotEmpty
                    ? widget.otherUserName[0].toUpperCase()
                    : '?',
                style: TextStyle(
                  fontWeight: FontWeight.w700,
                  color: cs.onPrimaryContainer,
                  fontSize: 14,
                ),
              ),
            ),
            const SizedBox(width: 10),
            Text(
              widget.otherUserName,
              style: const TextStyle(fontWeight: FontWeight.w700),
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          // ── Messages ───────────────────────────────────────────────────
          Expanded(
            child: BlocConsumer<ParentChatCubit, ParentChatState>(
              listener: (context, state) {
                if (state is ParentChatLoaded) _scrollToBottom();
              },
              builder: (context, state) {
                if (state is ParentChatLoading) {
                  return const Center(child: CircularProgressIndicator());
                }
                if (state is ParentChatError) {
                  return Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(state.message,
                            textAlign: TextAlign.center,
                            style: TextStyle(color: cs.error)),
                        const SizedBox(height: 12),
                        TextButton(
                          onPressed: () =>
                              context.read<ParentChatCubit>().load(),
                          child: const Text('Retry'),
                        ),
                      ],
                    ),
                  );
                }
                if (state is! ParentChatLoaded) {
                  return const SizedBox.shrink();
                }

                if (state.messages.isEmpty) {
                  return Center(
                    child: Text(
                      'No messages yet.\nSay hello!',
                      textAlign: TextAlign.center,
                      style: TextStyle(color: cs.onSurfaceVariant),
                    ),
                  );
                }

                return ListView.builder(
                  controller: _scrollCtrl,
                  padding: const EdgeInsets.fromLTRB(12, 16, 12, 8),
                  itemCount: state.messages.length,
                  itemBuilder: (context, i) {
                    final msg  = state.messages[i];
                    final mine = msg.isMine(myUserId);
                    return _ChatBubble(
                      body:      msg.body,
                      createdAt: msg.createdAt,
                      isMine:    mine,
                    );
                  },
                );
              },
            ),
          ),

          // ── Input bar ──────────────────────────────────────────────────
          _InputBar(
            controller: _ctrl,
            onSend: () => _send(context),
          ),
        ],
      ),
    );
  }

  Future<void> _send(BuildContext context) async {
    final body = _ctrl.text.trim();
    if (body.isEmpty) return;
    _ctrl.clear();
    final ok = await context.read<ParentChatCubit>().sendReply(body);
    if (ok) _scrollToBottom();
  }
}

// ── Chat bubble ───────────────────────────────────────────────────────────────

class _ChatBubble extends StatelessWidget {
  final String   body;
  final DateTime createdAt;
  final bool     isMine;

  const _ChatBubble({
    required this.body,
    required this.createdAt,
    required this.isMine,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;

    final bubbleColor = isMine ? cs.primary : cs.surfaceContainerHighest;
    final textColor   = isMine ? cs.onPrimary : cs.onSurface;
    final timeColor   = isMine
        ? cs.onPrimary.withValues(alpha: 0.7)
        : cs.onSurfaceVariant;

    return Align(
      alignment: isMine ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.72,
        ),
        margin: EdgeInsets.only(
          top: 3,
          bottom: 3,
          left:  isMine ? 48 : 0,
          right: isMine ? 0  : 48,
        ),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: bubbleColor,
          borderRadius: BorderRadius.only(
            topLeft:     const Radius.circular(18),
            topRight:    const Radius.circular(18),
            bottomLeft:  Radius.circular(isMine ? 18 : 4),
            bottomRight: Radius.circular(isMine ? 4  : 18),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              body,
              style: tt.bodyMedium?.copyWith(color: textColor, height: 1.4),
            ),
            const SizedBox(height: 4),
            Text(
              DateFormat('HH:mm').format(createdAt.toLocal()),
              style: TextStyle(fontSize: 10, color: timeColor),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Input bar ─────────────────────────────────────────────────────────────────

class _InputBar extends StatelessWidget {
  final TextEditingController controller;
  final VoidCallback onSend;

  const _InputBar({required this.controller, required this.onSend});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return BlocBuilder<ParentChatCubit, ParentChatState>(
      builder: (context, state) {
        final sending = state is ParentChatLoaded && state.sending;

        return Container(
          color: cs.surface,
          padding: EdgeInsets.only(
            left: 12,
            right: 8,
            top: 8,
            bottom: MediaQuery.of(context).viewInsets.bottom + 8,
          ),
          child: SafeArea(
            top: false,
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Expanded(
                  child: Container(
                    decoration: BoxDecoration(
                      color: cs.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(24),
                    ),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 4),
                    child: TextField(
                      controller: controller,
                      minLines: 1,
                      maxLines: 5,
                      textCapitalization: TextCapitalization.sentences,
                      decoration: const InputDecoration(
                        hintText: 'Write a message…',
                        border: InputBorder.none,
                        isDense: true,
                      ),
                      onSubmitted: (_) => onSend(),
                    ),
                  ),
                ),
                const SizedBox(width: 6),
                AnimatedSwitcher(
                  duration: const Duration(milliseconds: 200),
                  child: sending
                      ? Padding(
                          padding: const EdgeInsets.all(12),
                          child: SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(
                                strokeWidth: 2, color: cs.primary),
                          ),
                        )
                      : IconButton(
                          onPressed: onSend,
                          icon: Icon(Icons.send_rounded, color: cs.primary),
                          iconSize: 28,
                        ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
