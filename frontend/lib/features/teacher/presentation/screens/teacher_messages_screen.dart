import 'package:first_try/core/theme/theme.dart';
import 'package:first_try/core/widgets/shared/empty_state.dart';
import 'package:first_try/core/widgets/shared/error_view.dart';
import 'package:first_try/core/widgets/shared/loading_view.dart';
import 'package:first_try/core/widgets/shared/skeletons.dart';
import 'package:first_try/core/widgets/ui/ui.dart';
import 'package:first_try/features/teacher/data/models/teacher_extra_models.dart';
import 'package:first_try/features/teacher/presentation/cubit/teacher_messages_cubit.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:intl/intl.dart';

class TeacherMessagesScreen extends StatefulWidget {
  const TeacherMessagesScreen({super.key});

  @override
  State<TeacherMessagesScreen> createState() => _TeacherMessagesScreenState();
}

class _TeacherMessagesScreenState extends State<TeacherMessagesScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tab;

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 2, vsync: this);
    context.read<TeacherMessagesCubit>().load();
  }

  @override
  void dispose() {
    _tab.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Messages', style: TextStyle(fontWeight: FontWeight.w700)),
        bottom: TabBar(
          controller: _tab,
          tabs: [
            // Inbox tab — shows unread badge when there are unread messages
            BlocBuilder<TeacherMessagesCubit, TeacherMessagesState>(
              builder: (context, state) {
                final unread =
                    state is TeacherMessagesLoaded ? state.unreadCount : 0;
                return Tab(
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Text('Inbox'),
                      if (unread > 0) ...[
                        const SizedBox(width: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: cs.error,
                            borderRadius: Radii.pillRadius,
                          ),
                          child: Text(
                            '$unread',
                            style: TextStyle(
                              color: cs.onError,
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ],
                    ],
                  ),
                );
              },
            ),
            const Tab(text: 'Sent'),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _showCompose(context),
        icon: const Icon(Icons.edit_rounded),
        label: const Text('Compose'),
      ),
      body: BlocConsumer<TeacherMessagesCubit, TeacherMessagesState>(
        listener: (context, state) {
          if (state is TeacherMessagesError) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(state.message),
                backgroundColor: Theme.of(context).colorScheme.error,
              ),
            );
          }
        },
        builder: (context, state) {
          if (state is TeacherMessagesLoading ||
              state is TeacherMessagesInitial) {
            return const CardListSkeleton(cardHeight: 80);
          }
          if (state is TeacherMessagesError) {
            return ErrorView(
              message: state.message,
              onRetry: () => context.read<TeacherMessagesCubit>().load(),
            );
          }
          if (state is! TeacherMessagesLoaded) return const SizedBox.shrink();

          return TabBarView(
            controller: _tab,
            children: [
              _MessageList(messages: state.inbox, isInbox: true),
              _MessageList(messages: state.sent,  isInbox: false),
            ],
          );
        },
      ),
    );
  }

  void _showCompose(BuildContext context) {
    final cubit = context.read<TeacherMessagesCubit>();
    cubit.loadParents(); // pre-fetch parents so the dropdown is ready
    showAppBottomSheet<void>(
      context: context,
      title: 'New Message',
      subtitle: 'Send a message to a parent.',
      builder: (ctx) => BlocProvider.value(
        value: cubit,
        child: const _ComposeSheet(),
      ),
    );
  }
}

// ── Message list ──────────────────────────────────────────────────────────────

class _MessageList extends StatelessWidget {
  final List<TeacherMessageModel> messages;
  final bool isInbox;
  const _MessageList({required this.messages, required this.isInbox});

  @override
  Widget build(BuildContext context) {
    if (messages.isEmpty) {
      return EmptyState(
        icon: Icons.mail_outline_rounded,
        title: isInbox ? 'No messages' : 'No sent messages',
        subtitle: isInbox
            ? "You'll see messages from parents here."
            : 'Your sent messages will appear here.',
      );
    }
    return RefreshIndicator(
      onRefresh: () => context.read<TeacherMessagesCubit>().load(),
      child: ListView.separated(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
        itemCount: messages.length,
        separatorBuilder: (_, _) => const SizedBox(height: 10),
        itemBuilder: (context, i) =>
            _MessageCard(message: messages[i], isInbox: isInbox),
      ),
    );
  }
}

// ── Message card ──────────────────────────────────────────────────────────────

class _MessageCard extends StatelessWidget {
  final TeacherMessageModel message;
  final bool isInbox;
  const _MessageCard({required this.message, required this.isInbox});

  @override
  Widget build(BuildContext context) {
    final cs       = Theme.of(context).colorScheme;
    final tt       = Theme.of(context).textTheme;
    final isUnread = isInbox && !message.isRead;
    final other    = isInbox
        ? (message.senderName   ?? 'Parent')
        : (message.receiverName ?? 'Parent');
    final initial  = other.isNotEmpty ? other[0].toUpperCase() : '?';

    return AppCard.surface(
      onTap: () => _open(context),
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Header: avatar · name · status pill · time ──────────────
          Row(
            children: [
              // Avatar — Container + BoxDecoration + Radii pattern
              Stack(
                clipBehavior: Clip.none,
                children: [
                  Container(
                    width: 38,
                    height: 38,
                    decoration: BoxDecoration(
                      color: isUnread
                          ? cs.primaryContainer
                          : cs.surfaceContainerHighest,
                      borderRadius: Radii.smRadius,
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      initial,
                      style: tt.titleSmall?.copyWith(
                        fontWeight: FontWeight.w800,
                        color: isUnread
                            ? cs.onPrimaryContainer
                            : cs.onSurfaceVariant,
                      ),
                    ),
                  ),
                  if (isUnread)
                    Positioned(
                      right: -3,
                      top: -3,
                      child: Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          color: cs.primary,
                          shape: BoxShape.circle,
                          border: Border.all(color: cs.surface, width: 1.5),
                        ),
                      ),
                    ),
                ],
              ),
              const SizedBox(width: 10),

              // Name
              Expanded(
                child: Text(
                  other,
                  style: tt.bodyMedium?.copyWith(
                    fontWeight: isUnread ? FontWeight.w700 : FontWeight.w600,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 8),

              // Status pill
              if (!isInbox && message.isRead)
                StatusPill(
                  label: 'Read',
                  tone: StatusTone.success,
                  icon: Icons.done_all_rounded,
                  dense: true,
                )
              else if (isUnread)
                StatusPill(
                  label: 'New',
                  tone: StatusTone.primary,
                  icon: Icons.mark_email_unread_outlined,
                  dense: true,
                ),

              const SizedBox(width: 8),
              Text(
                _fmt(message.createdAt),
                style: tt.labelSmall?.copyWith(
                  color: isUnread ? cs.primary : cs.onSurfaceVariant,
                  fontWeight: isUnread ? FontWeight.w700 : FontWeight.normal,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),

          // ── Subject ─────────────────────────────────────────────────
          if (message.subject != null) ...[
            Text(
              message.subject!,
              style: tt.bodyMedium?.copyWith(
                fontWeight: isUnread ? FontWeight.w700 : FontWeight.w600,
              ),
              overflow: TextOverflow.ellipsis,
            ),
            const SizedBox(height: 3),
          ],

          // ── Body preview ─────────────────────────────────────────────
          Text(
            message.body,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
          ),
          const SizedBox(height: 8),

          // ── Footer: student · time ────────────────────────────────────
          Row(
            children: [
              if (message.studentName != null) ...[
                Icon(Icons.person_outline_rounded,
                    size: 13, color: cs.primary),
                const SizedBox(width: 4),
                Text(
                  message.studentName!,
                  style: tt.labelSmall?.copyWith(
                    color: cs.primary,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(width: 10),
              ],
              Icon(Icons.access_time_rounded,
                  size: 13, color: cs.onSurfaceVariant),
              const SizedBox(width: 4),
              Text(
                _fmtFull(message.createdAt),
                style: tt.labelSmall?.copyWith(color: cs.onSurfaceVariant),
              ),
              const Spacer(),
              Icon(Icons.chevron_right_rounded,
                  size: 16, color: cs.onSurfaceVariant),
            ],
          ),
        ],
      ),
    );
  }

  void _open(BuildContext context) {
    final cubit = context.read<TeacherMessagesCubit>();
    cubit.open(message.id);
    showAppBottomSheet<void>(
      context: context,
      title: message.subject ?? 'Message',
      builder: (_) => BlocProvider.value(
        value: cubit,
        child: const _DetailSheet(),
      ),
    ).then((_) => cubit.closeOpened());
  }

  String _fmt(String iso) {
    try {
      final dt   = DateTime.parse(iso).toLocal();
      final now  = DateTime.now();
      final diff = now.difference(dt);
      if (diff.inMinutes < 1) return 'Just now';
      if (diff.inHours  < 1) return '${diff.inMinutes}m ago';
      if (diff.inDays   < 1) return '${diff.inHours}h ago';
      if (diff.inDays   < 7) return '${diff.inDays}d ago';
      return DateFormat('d MMM').format(dt);
    } catch (_) {
      return '';
    }
  }

  String _fmtFull(String iso) {
    try {
      return DateFormat('d MMM y').format(DateTime.parse(iso).toLocal());
    } catch (_) {
      return iso;
    }
  }
}

// ── Detail bottom-sheet ───────────────────────────────────────────────────────

class _DetailSheet extends StatelessWidget {
  const _DetailSheet();

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<TeacherMessagesCubit, TeacherMessagesState>(
      builder: (context, state) {
        if (state is! TeacherMessagesLoaded || state.opened == null) {
          return const Padding(
            padding: EdgeInsets.all(40),
            child: LoadingView(),
          );
        }
        final m  = state.opened!;
        final cs = Theme.of(context).colorScheme;
        final tt = Theme.of(context).textTheme;

        return SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── Metadata ────────────────────────────────────────────
              _MetaRow(
                icon: Icons.person_outline_rounded,
                label: 'From',
                value: m.senderName ?? '—',
              ),
              const SizedBox(height: 4),
              _MetaRow(
                icon: Icons.person_rounded,
                label: 'To',
                value: m.receiverName ?? '—',
              ),
              if (m.studentName != null) ...[
                const SizedBox(height: 4),
                _MetaRow(
                  icon: Icons.child_care_rounded,
                  label: 'Re',
                  value: m.studentName!,
                ),
              ],
              const SizedBox(height: 4),
              _MetaRow(
                icon: Icons.access_time_rounded,
                label: 'Sent',
                value: _fmtFull(m.createdAt),
              ),
              if (m.readAt != null) ...[
                const SizedBox(height: 4),
                _MetaRow(
                  icon: Icons.done_all_rounded,
                  label: 'Read',
                  value: _fmtFull(m.readAt!),
                  valueColor: cs.primary,
                ),
              ],
              const Divider(height: 24),

              // ── Body ────────────────────────────────────────────────
              Text(
                m.body,
                style: tt.bodyMedium?.copyWith(height: 1.6, color: cs.onSurface),
              ),
              const SizedBox(height: 16),
            ],
          ),
        );
      },
    );
  }

  String _fmtFull(String iso) {
    try {
      return DateFormat('d MMM yyyy, HH:mm').format(DateTime.parse(iso).toLocal());
    } catch (_) {
      return iso;
    }
  }
}

class _MetaRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color? valueColor;
  const _MetaRow({
    required this.icon,
    required this.label,
    required this.value,
    this.valueColor,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 14, color: cs.onSurfaceVariant),
        const SizedBox(width: 6),
        Text(
          '$label: ',
          style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
        ),
        Expanded(
          child: Text(
            value,
            style: tt.bodySmall?.copyWith(
              color: valueColor ?? cs.onSurface,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ],
    );
  }
}

// ── Compose sheet ─────────────────────────────────────────────────────────────
//
// Cascading: Parent → Student (optional, filtered by parent) → Subject → Body

class _ComposeSheet extends StatefulWidget {
  const _ComposeSheet();

  @override
  State<_ComposeSheet> createState() => _ComposeSheetState();
}

class _ComposeSheetState extends State<_ComposeSheet> {
  final _subjectCtl = TextEditingController();
  final _bodyCtl    = TextEditingController();

  int?    _parentUserId; // required — User.id of the selected parent
  int?    _studentId;    // optional
  String? _subject;      // optional

  @override
  void dispose() {
    _subjectCtl.dispose();
    _bodyCtl.dispose();
    super.dispose();
  }

  void _onParentChanged(int? userId) {
    setState(() {
      _parentUserId = userId;
      _studentId    = null; // reset child when parent changes
    });
  }

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<TeacherMessagesCubit, TeacherMessagesState>(
      builder: (context, state) {
        final loaded  = state is TeacherMessagesLoaded ? state : null;
        final sending = loaded?.sending ?? false;
        final parentsLoading = loaded?.parentsLoading ?? false;
        final parents = loaded?.parents ?? [];

        // Students for the selected parent
        final selectedParent = parents
            .where((p) => p.userId == _parentUserId)
            .firstOrNull;
        final students = selectedParent?.students ?? [];

        return SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 4, 20, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [

              // ── Parent ───────────────────────────────────────────────
              DropdownButtonFormField<int?>(
                initialValue: _parentUserId,
                decoration: InputDecoration(
                  labelText: 'Parent *',
                  prefixIcon: parentsLoading
                      ? const Padding(
                          padding: EdgeInsets.all(12),
                          child: SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                        )
                      : const Icon(Icons.person_rounded),
                ),
                hint: parentsLoading
                    ? const Text('Loading parents…')
                    : const Text('Select a parent'),
                items: parents
                    .map((p) => DropdownMenuItem<int?>(
                        value: p.userId, child: Text(p.name)))
                    .toList(),
                onChanged: (parentsLoading || parents.isEmpty)
                    ? null
                    : _onParentChanged,
              ),
              const SizedBox(height: 14),

              // ── Student (optional, filtered to the selected parent's children)
              DropdownButtonFormField<int?>(
                initialValue: _studentId,
                decoration: const InputDecoration(
                  labelText: 'Student (optional)',
                  prefixIcon: Icon(Icons.child_care_rounded),
                ),
                hint: _parentUserId == null
                    ? const Text('Select a parent first')
                    : students.isEmpty
                        ? const Text('No students found')
                        : const Text('All students'),
                items: [
                  if (_parentUserId != null)
                    const DropdownMenuItem<int?>(
                        value: null, child: Text('All students')),
                  for (final s in students)
                    DropdownMenuItem<int?>(value: s.id, child: Text(s.name)),
                ],
                onChanged: (_parentUserId == null || students.isEmpty)
                    ? null
                    : (v) => setState(() => _studentId = v),
              ),
              const SizedBox(height: 14),

              // ── Subject (optional) ───────────────────────────────────
              TextField(
                controller: _subjectCtl,
                enabled: _parentUserId != null,
                textCapitalization: TextCapitalization.sentences,
                decoration: const InputDecoration(
                  labelText: 'Subject (optional)',
                  prefixIcon: Icon(Icons.subject_rounded),
                  hintText: 'e.g. Homework update',
                ),
                onChanged: (v) => setState(
                    () => _subject = v.trim().isEmpty ? null : v.trim()),
              ),
              const SizedBox(height: 14),

              // ── Body ─────────────────────────────────────────────────
              TextField(
                controller: _bodyCtl,
                minLines: 5,
                maxLines: 10,
                enabled: _parentUserId != null,
                textCapitalization: TextCapitalization.sentences,
                decoration: const InputDecoration(
                  labelText: 'Message *',
                  alignLabelWithHint: true,
                  prefixIcon: Padding(
                    padding: EdgeInsets.only(bottom: 80),
                    child: Icon(Icons.notes_rounded),
                  ),
                ),
              ),
              const SizedBox(height: 20),

              // ── Send ─────────────────────────────────────────────────
              AppButton.primary(
                label: 'Send Message',
                icon: Icons.send_rounded,
                fullWidth: true,
                size: AppButtonSize.lg,
                loading: sending,
                onPressed: sending ? null : () => _send(context),
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> _send(BuildContext context) async {
    final body = _bodyCtl.text.trim();

    if (_parentUserId == null) {
      _snack(context, 'Please select a parent.');
      return;
    }
    if (body.isEmpty) {
      _snack(context, 'Please write a message.');
      return;
    }

    final ok = await context.read<TeacherMessagesCubit>().send(
          receiverUserId: _parentUserId!,
          studentId: _studentId,
          subject: _subject,
          body: body,
        );

    if (!context.mounted) return;
    if (ok) {
      Navigator.pop(context);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Message sent.')),
      );
    }
  }

  void _snack(BuildContext context, String msg) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
}
