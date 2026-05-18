<?php

namespace App\Http\Requests\StopEvent;

use Illuminate\Foundation\Http\FormRequest;

class StoreStopEventRequest extends FormRequest
{
    public function authorize(): bool { return true; }

    public function rules(): array
    {
        return [
            'student_id' => 'required|integer|exists:students,id',
            'stop_id'    => 'required|integer|exists:routestop,stop_id',
            'eventtype'  => 'required|in:boarded,dropped',
            'eventat'    => 'nullable|date',
        ];
    }
}
