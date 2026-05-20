import 'package:first_try/core/theme/theme.dart';
import 'package:first_try/core/widgets/shared/empty_state.dart';
import 'package:first_try/core/widgets/shared/error_view.dart';
import 'package:first_try/core/widgets/shared/skeletons.dart';
import 'package:first_try/core/widgets/ui/ui.dart';
import 'package:first_try/features/auth/current_user.dart';
import 'package:first_try/features/parent/data/models/parent_extra_models.dart';
import 'package:first_try/features/parent/data/models/parent_models.dart';
import 'package:first_try/features/parent/presentation/cubit/messages_cubit.dart';
import 'package:first_try/features/parent/presentation/cubit/parent_cubit.dart';
import 'package:first_try/features/parent/presentation/cubit/parent_state.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:intl/intl.dart';

class ParentMessagesScreen extends StatefulWidget {
  const ParentMessagesScreen({super.key});

  @override
  State<ParentMessagesScreen> createState() => _ParentMessagesScreenState();
}

class _ParentMessagesScreenState extends State<ParentMessagesScreen> {
  @override
  void initState() {
    super.initState();
    context.read<MessagesCubit>().loadInbox();
  }

  @override
  Widget build(BuildContext context) {
    final myUserId = context.currentUserId;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Messages', style: TextStyle(fontWeight: FontWeight.w700)),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _showCompose(context),
        icon: const Icon(Icons.edit_rounded),
        label: const Text('New Message'),
      ),
      body: BlocConsumer<MessagesCubit, MessagesState>(
        listener: (context, state) {
          if (state is MessagesError) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(state.message),
                backgroundColor: Theme.of(context).colorScheme.error,
              ),
            );
          }
        },
        builder: (context, state) {
          if (state is MessagesInitial || state is MessagesLoading) {
            return const CardListSkeleton(cardHeight: 80);
          }
          if (state is MessagesError) {
            return ErrorView(
              message: state.message,
              onRetry: () => context.read<MessagesCubit>().loadInbox(),
            );
          }
          if (state is! MessagesLoaded) return const SizedBox.shrink();

          if (state.inbox.isEmpty) {
            return const EmptyState(
              icon: Icons.mail_outline_rounded,
              title: 'No messages yet',
              subtitle: 'Tap "New Message" to contact a teacher.',
            );
          }

          return RefreshIndicator(
            onRefresh: () => context.read<MessagesCubit>().loadInbox(),
            child: ListView.separated(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
              itemCount: state.inbox.length,
              separatorBuilder: (_, _) => const SizedBox(height: 10),
              itemBuilder: (context, i) => _MessageCard(
                message: state.inbox[i],
                myUserId: myUserId,
                onTap: () => _showDetail(context, state.inbox[i]),
              ),
            ),
          );
        },
      ),
    );
  }

  void _showCompose(BuildContext context) {
    final cubit       = context.read<MessagesCubit>();
    final parentCubit = context.read<ParentCubit>();

    // Pre-fetch all teachers so the dropdown is ready if no child is chosen.
    cubit.loadTeachers();

    showAppBottomSheet<void>(
      context: context,
      title: 'New Message',
      subtitle: "Send a message to one of your child's teachers.",
      builder: (ctx) => MultiBlocProvider(
        providers: [
          BlocProvider.value(value: cubit),
          BlocProvider.value(value: parentCubit),
        ],
        child: const _ComposeSheet(),
      ),
    );
  }

  void _showDetail(BuildContext context, MessageModel message) {
    final cubit = context.read<MessagesCubit>();
    cubit.open(message.id);

    showAppBottomSheet<void>(
      context: context,
      title: message.subject,
      builder: (ctx) => BlocProvider.value(
        value: cubit,
        child: _DetailSheet(message: message),
      ),
    );
  }
}

// ── Message card ──────────────────────────────────────────────────────────────

class _MessageCard extends StatelessWidget {
  final MessageModel message;
  final int myUserId;
  final VoidCallback onTap;

  const _MessageCard({
    required this.message,
    required this.myUserId,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final cs       = Theme.of(context).colorScheme;
    final tt       = Theme.of(context).textTheme;
    final isSent   = message.sentByMe(myUserId);
    final isUnread = !isSent && !message.isRead;

    // Show the other party's name (teacher when sent, sender when received).
    final otherName = isSent
        ? (message.receiverName ?? 'Teacher')
        : (message.senderName  ?? 'Teacher');
    final initial = otherName.isNotEmpty ? otherName[0].toUpperCase() : '?';

    return AppCard.surface(
      onTap: onTap,
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Header row: avatar · name · pill · time ─────────────────
          Row(
            children: [
              // Avatar — follows the Container + BoxDecoration + Radii pattern
              Stack(
                clipBehavior: Clip.none,
                children: [
                  Container(
                    width: 38,
                    height: 38,
                    decoration: BoxDecoration(
                      color: cs.primaryContainer,
                      borderRadius: Radii.smRadius,
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      initial,
                      style: tt.titleSmall?.copyWith(
                        fontWeight: FontWeight.w800,
                        color: cs.onPrimaryContainer,
                      ),
                    ),
                  ),
                  // Unread dot (received only)
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
                  otherName,
                  style: tt.bodyMedium?.copyWith(
                    fontWeight: isUnread ? FontWeight.w700 : FontWeight.w600,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 8),

              // Sent / Unread pill
              if (isSent)
                StatusPill(
                  label: 'Sent',
                  tone: StatusTone.neutral,
                  icon: Icons.arrow_upward_rounded,
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
          Text(
            message.subject,
            style: tt.bodyMedium?.copyWith(
              fontWeight: isUnread ? FontWeight.w700 : FontWeight.normal,
            ),
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 3),

          // ── Body preview ─────────────────────────────────────────────
          Text(
            message.body,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant),
          ),
          const SizedBox(height: 8),

          // ── Footer: student · chevron ────────────────────────────────
          Row(
            children: [
              if (message.studentName != null) ...[
                Icon(Icons.person_outline_rounded, size: 13, color: cs.onSurfaceVariant),
                const SizedBox(width: 4),
                Text(
                  message.studentName!,
                  style: tt.labelSmall?.copyWith(color: cs.onSurfaceVariant),
                ),
                const SizedBox(width: 10),
              ],
              Icon(Icons.access_time_rounded, size: 13, color: cs.onSurfaceVariant),
              const SizedBox(width: 4),
              Text(
                _fmtFull(message.createdAt),
                style: tt.labelSmall?.copyWith(color: cs.onSurfaceVariant),
              ),
              const Spacer(),
              Icon(Icons.chevron_right_rounded, size: 16, color: cs.onSurfaceVariant),
            ],
          ),
        ],
      ),
    );
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
  final MessageModel message;
  const _DetailSheet({required this.message});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // ── Metadata ────────────────────────────────────────────────
          _MetaRow(icon: Icons.person_outline_rounded, label: 'From', value: message.senderName   ?? '—'),
          const SizedBox(height: 4),
          _MetaRow(icon: Icons.person_rounded,         label: 'To',   value: message.receiverName ?? '—'),
          if (message.studentName != null) ...[
            const SizedBox(height: 4),
            _MetaRow(icon: Icons.child_care_rounded, label: 'Re', value: message.studentName!),
          ],
          const SizedBox(height: 4),
          _MetaRow(icon: Icons.access_time_rounded, label: 'Sent', value: _fmtFull(message.createdAt)),
          const Divider(height: 24),

          // ── Body ────────────────────────────────────────────────────
          Text(
            message.body,
            style: tt.bodyMedium?.copyWith(height: 1.6, color: cs.onSurface),
          ),
          const SizedBox(height: 16),
        ],
      ),
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
  const _MetaRow({required this.icon, required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 14, color: cs.onSurfaceVariant),
        const SizedBox(width: 6),
        Text('$label: ', style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant)),
        Expanded(
          child: Text(
            value,
            style: tt.bodySmall?.copyWith(
              color: cs.onSurface,
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
// Cascading selection: Child (optional) → Teacher → Subject → Body

class _ComposeSheet extends StatefulWidget {
  const _ComposeSheet();

  @override
  State<_ComposeSheet> createState() => _ComposeSheetState();
}

class _ComposeSheetState extends State<_ComposeSheet> {
  final _subjectCtl = TextEditingController(); // free-text when no child picked
  final _bodyCtl    = TextEditingController();

  int?    _studentId; // optional
  int?    _teacherId; // required
  String? _subject;   // required

  @override
  void dispose() {
    _subjectCtl.dispose();
    _bodyCtl.dispose();
    super.dispose();
  }

  void _onChildChanged(int? childId) {
    setState(() {
      _studentId = childId;
      _teacherId = null;
      _subject   = null;
      _subjectCtl.clear();
    });
    if (childId != null) {
      context.read<MessagesCubit>().loadTeachersForChild(childId);
    }
  }

  void _onTeacherChanged(int? teacherId) {
    setState(() {
      _teacherId = teacherId;
      _subject   = null;
      _subjectCtl.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<MessagesCubit, MessagesState>(
      builder: (context, msgState) {
        return BlocBuilder<ParentCubit, ParentState>(
          builder: (context, parentState) {
            final loaded  = msgState is MessagesLoaded ? msgState : null;
            final sending = loaded?.sending ?? false;

            final children = parentState is ParentLoaded
                ? parentState.profile.children
                : <ChildSummaryModel>[];

            // Teacher list depends on whether a child is selected
            final bool childPicked      = _studentId != null;
            final bool teachersLoading  = childPicked
                ? (loaded?.isLoadingTeachersFor(_studentId!) ?? false)
                : (loaded?.allTeachersLoading ?? false);
            final List<TeacherSummaryModel> teachers = childPicked
                ? (loaded?.teachersFor(_studentId!) ?? [])
                : (loaded?.allTeachers ?? []);

            // Subject dropdown only when child + teacher are both chosen
            final bool useSubjectDropdown = childPicked && _teacherId != null;
            final List<String> subjects   = useSubjectDropdown
                ? (teachers.where((t) => t.id == _teacherId).firstOrNull?.subjects ?? [])
                : [];

            return SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 4, 20, 24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [

                  // ── Child (optional) ────────────────────────────────
                  DropdownButtonFormField<int?>(
                    initialValue: _studentId,
                    decoration: const InputDecoration(
                      labelText: 'Child (optional)',
                      prefixIcon: Icon(Icons.child_care_rounded),
                    ),
                    hint: const Text('None — general message'),
                    items: [
                      const DropdownMenuItem<int?>(
                          value: null, child: Text('None — general message')),
                      for (final c in children)
                        DropdownMenuItem<int?>(value: c.id, child: Text(c.name)),
                    ],
                    onChanged: _onChildChanged,
                  ),
                  const SizedBox(height: 14),

                  // ── Teacher ──────────────────────────────────────────
                  DropdownButtonFormField<int?>(
                    initialValue: _teacherId,
                    decoration: InputDecoration(
                      labelText: 'Teacher *',
                      prefixIcon: teachersLoading
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
                    hint: teachersLoading
                        ? const Text('Loading teachers…')
                        : const Text('Select a teacher'),
                    items: teachers
                        .map((t) => DropdownMenuItem<int?>(
                            value: t.id, child: Text(t.name)))
                        .toList(),
                    onChanged: (teachersLoading || teachers.isEmpty)
                        ? null
                        : _onTeacherChanged,
                  ),
                  const SizedBox(height: 14),

                  // ── Subject ──────────────────────────────────────────
                  // Dropdown when child + teacher chosen; free-text otherwise
                  if (useSubjectDropdown)
                    DropdownButtonFormField<String?>(
                      initialValue: _subject,
                      decoration: const InputDecoration(
                        labelText: 'Subject *',
                        prefixIcon: Icon(Icons.subject_rounded),
                      ),
                      hint: subjects.isEmpty
                          ? const Text('No subjects found')
                          : const Text('Select a subject'),
                      items: subjects
                          .map((s) => DropdownMenuItem<String?>(
                              value: s, child: Text(s)))
                          .toList(),
                      onChanged: subjects.isEmpty
                          ? null
                          : (v) => setState(() => _subject = v),
                    )
                  else
                    TextField(
                      controller: _subjectCtl,
                      enabled: _teacherId != null,
                      textCapitalization: TextCapitalization.sentences,
                      decoration: const InputDecoration(
                        labelText: 'Subject *',
                        prefixIcon: Icon(Icons.subject_rounded),
                        hintText: 'e.g. Homework question',
                      ),
                      onChanged: (v) => setState(
                          () => _subject = v.trim().isEmpty ? null : v.trim()),
                    ),
                  const SizedBox(height: 14),

                  // ── Message body ──────────────────────────────────────
                  TextField(
                    controller: _bodyCtl,
                    minLines: 5,
                    maxLines: 10,
                    enabled: _teacherId != null,
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

                  // ── Send ─────────────────────────────────────────────
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
      },
    );
  }

  Future<void> _send(BuildContext context) async {
    final body = _bodyCtl.text.trim();

    if (_teacherId == null) {
      _snack(context, 'Please select a teacher.');
      return;
    }
    if (_subject == null || _subject!.isEmpty) {
      _snack(context, 'Please enter a subject.');
      return;
    }
    if (body.isEmpty) {
      _snack(context, 'Please write a message.');
      return;
    }

    final ok = await context.read<MessagesCubit>().send(
          teacherId: _teacherId!,
          studentId: _studentId,
          subject: _subject!,
          body: body,
        );

    if (!context.mounted) return;
    if (ok) {
      Navigator.pop(context);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Message sent successfully.')),
      );
    }
  }

  void _snack(BuildContext context, String msg) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
}
